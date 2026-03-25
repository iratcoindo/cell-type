from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage import measure, morphology, segmentation, feature
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="iRATco Cell Counter",
    page_icon="logo.png",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================
col1, col2 = st.columns([8, 2])
with col1:
    st.title("iRATco Cell Counter")
    st.markdown(
        "<span style='font-size:16px;color:gray;'>**version 1.1.0**</span>",
        unsafe_allow_html=True
    )
with col2:
    st.image("logo_iratco.png", width=250)

st.caption("Semi-automatic inflammatory cell counting from histopathology images")

# =========================================================
# SESSION STATE
# =========================================================
ANALYSIS_CLASSES = [
    "Lymphocyte",
    "Macrophage",
    "Neutrophil",
    "Eosinophil",
    "Basophil",
    "Plasma Cell"
]

ALL_CLASSES = ANALYSIS_CLASSES + ["Unknown"]

if "samples" not in st.session_state:
    st.session_state.samples = {cls: [] for cls in ALL_CLASSES}

if "objects_df" not in st.session_state:
    st.session_state.objects_df = None

if "labeled_mask" not in st.session_state:
    st.session_state.labeled_mask = None

if "preview_rgb" not in st.session_state:
    st.session_state.preview_rgb = None

if "result_df" not in st.session_state:
    st.session_state.result_df = None

if "trained" not in st.session_state:
    st.session_state.trained = False

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

for cls in ALL_CLASSES:
    if cls not in st.session_state.samples:
        st.session_state.samples[cls] = []

# =========================================================
# HELPERS
# =========================================================
def reset_all_annotations():
    st.session_state.samples = {cls: [] for cls in ALL_CLASSES}
    st.session_state.result_df = None
    st.session_state.trained = False


def pil_to_rgb_array(pil_img):
    return np.array(pil_img.convert("RGB"))


def make_display_image(rgb, max_width=700):
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb.copy(), 1.0
    scale = max_width / w
    new_h = int(h * scale)
    resized = cv2.resize(rgb, (max_width, new_h))
    return resized, scale


def preprocess_and_segment(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    binary_bool = binary > 0
    binary_bool = morphology.remove_small_objects(binary_bool, min_size=40)
    binary_bool = morphology.remove_small_holes(binary_bool, area_threshold=40)

    dist = cv2.distanceTransform((binary_bool.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    local_max = feature.peak_local_max(
        dist_norm,
        min_distance=8,
        labels=binary_bool
    )

    markers = np.zeros(binary_bool.shape, dtype=np.int32)
    for i, (r, c) in enumerate(local_max, start=1):
        markers[r, c] = i

    markers = morphology.dilation(markers, morphology.disk(2))
    labeled = segmentation.watershed(-dist, markers, mask=binary_bool)

    return gray, binary_bool, labeled


def extract_object_features(rgb, gray, labeled):
    records = []
    props = measure.regionprops(labeled, intensity_image=gray)

    for prop in props:
        area = prop.area
        if area < 40:
            continue

        minr, minc, maxr, maxc = prop.bbox
        perimeter = prop.perimeter if prop.perimeter > 0 else np.nan
        major_axis = prop.major_axis_length if prop.major_axis_length > 0 else np.nan

        circularity = (
            4 * np.pi * area / (perimeter ** 2)
            if perimeter and perimeter > 0 else np.nan
        )
        roundness = (
            4 * area / (np.pi * (major_axis ** 2))
            if major_axis and major_axis > 0 else np.nan
        )

        mask = labeled[minr:maxr, minc:maxc] == prop.label
        gray_patch = gray[minr:maxr, minc:maxc]
        rgb_patch = rgb[minr:maxr, minc:maxc]

        if np.sum(mask) == 0:
            continue

        pix_gray = gray_patch[mask]
        pix_rgb = rgb_patch[mask]

        mean_intensity = float(np.mean(pix_gray))
        std_intensity = float(np.std(pix_gray))

        mean_r = float(np.mean(pix_rgb[:, 0]))
        mean_g = float(np.mean(pix_rgb[:, 1]))
        mean_b = float(np.mean(pix_rgb[:, 2]))

        lap_var = float(np.var(cv2.Laplacian(gray_patch, cv2.CV_64F)))

        eccentricity = float(prop.eccentricity) if prop.eccentricity is not None else np.nan
        solidity = float(prop.solidity) if prop.solidity is not None else np.nan

        cy, cx = prop.centroid

        records.append({
            "label_id": prop.label,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "bbox_minr": int(minr),
            "bbox_minc": int(minc),
            "bbox_maxr": int(maxr),
            "bbox_maxc": int(maxc),
            "area": float(area),
            "perimeter": float(perimeter) if not np.isnan(perimeter) else np.nan,
            "major_axis_length": float(major_axis) if not np.isnan(major_axis) else np.nan,
            "circularity": float(circularity) if not np.isnan(circularity) else np.nan,
            "roundness": float(roundness) if not np.isnan(roundness) else np.nan,
            "eccentricity": eccentricity,
            "solidity": solidity,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "granularity": lap_var
        })

    return pd.DataFrame(records)


def find_nearest_object(x, y, objects_df, max_dist=30):
    if objects_df is None or objects_df.empty:
        return None

    dx = objects_df["centroid_x"] - x
    dy = objects_df["centroid_y"] - y
    dist = np.sqrt(dx ** 2 + dy ** 2)

    idx = dist.idxmin()
    if dist.loc[idx] <= max_dist:
        return int(objects_df.loc[idx, "label_id"])
    return None


def build_training_table(objects_df, samples_dict):
    rows = []

    for cls_name, label_ids in samples_dict.items():
        for lid in label_ids:
            row = objects_df[objects_df["label_id"] == lid]
            if len(row) == 1:
                r = row.iloc[0].copy()
                r["target_class"] = cls_name
                rows.append(r)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def annotate_image(rgb, objects_df, result_df=None, sample_dict=None):
    out = rgb.copy()

    color_map = {
        "Lymphocyte": (255, 0, 0),
        "Macrophage": (0, 255, 0),
        "Neutrophil": (0, 0, 255),
        "Eosinophil": (255, 165, 0),
        "Basophil": (128, 0, 255),
        "Plasma Cell": (0, 165, 255),
        "Unknown": (180, 180, 180)
    }

    if result_df is not None and not result_df.empty:
        for _, row in result_df.iterrows():
            cls = row["predicted_class"]
            color_val = color_map.get(cls, (255, 255, 255))
            minr = int(row["bbox_minr"])
            minc = int(row["bbox_minc"])
            maxr = int(row["bbox_maxr"])
            maxc = int(row["bbox_maxc"])
            cv2.rectangle(out, (minc, minr), (maxc, maxr), color_val, 1)

    if sample_dict is not None and objects_df is not None:
        for cls, label_ids in sample_dict.items():
            color_val = color_map.get(cls, (255, 255, 255))
            for lid in label_ids:
                row = objects_df[objects_df["label_id"] == lid]
                if len(row) == 1:
                    cx = int(row.iloc[0]["centroid_x"])
                    cy = int(row.iloc[0]["centroid_y"])
                    cv2.circle(out, (cx, cy), 7, color_val, 2)
                    cv2.putText(
                        out, cls[:3], (cx + 4, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_val, 1, cv2.LINE_AA
                    )
    return out


def feature_columns():
    return [
        "area",
        "perimeter",
        "major_axis_length",
        "circularity",
        "roundness",
        "eccentricity",
        "solidity",
        "mean_intensity",
        "std_intensity",
        "mean_r",
        "mean_g",
        "mean_b",
        "granularity"
    ]


def make_colored_segmentation(rgb, labeled, result_df=None, sample_dict=None, objects_df=None):
    seg_rgb = np.zeros_like(rgb, dtype=np.uint8)

    class_color_map = {
        "Lymphocyte": (255, 0, 0),
        "Macrophage": (0, 255, 0),
        "Neutrophil": (0, 0, 255),
        "Eosinophil": (255, 165, 0),
        "Basophil": (128, 0, 255),
        "Plasma Cell": (0, 165, 255),
        "Unknown": (180, 180, 180)
    }

    if result_df is not None and not result_df.empty and "predicted_class" in result_df.columns:
        for _, row in result_df.iterrows():
            lid = int(row["label_id"])
            cls = row["predicted_class"]
            color_val = class_color_map.get(cls, (255, 255, 255))
            seg_rgb[labeled == lid] = color_val
    else:
        rng = np.random.default_rng(42)
        unique_labels = np.unique(labeled)
        for lid in unique_labels:
            if lid == 0:
                continue
            rand_color = rng.integers(50, 255, size=3, dtype=np.uint8)
            seg_rgb[labeled == lid] = rand_color

    boundaries = segmentation.find_boundaries(labeled, mode="outer")
    seg_rgb[boundaries] = [255, 255, 255]

    if sample_dict is not None and objects_df is not None:
        for cls, label_ids in sample_dict.items():
            draw_color = class_color_map.get(cls, (255, 255, 255))
            for lid in label_ids:
                row = objects_df[objects_df["label_id"] == lid]
                if len(row) == 1:
                    cx = int(row.iloc[0]["centroid_x"])
                    cy = int(row.iloc[0]["centroid_y"])
                    cv2.circle(seg_rgb, (cx, cy), 7, draw_color, 2)
                    cv2.putText(
                        seg_rgb, cls[:3], (cx + 4, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, draw_color, 1, cv2.LINE_AA
                    )

    return seg_rgb


def crop_object_thumbnail(rgb, labeled, row, pad=8, target_size=96):
    minr = max(int(row["bbox_minr"]) - pad, 0)
    minc = max(int(row["bbox_minc"]) - pad, 0)
    maxr = min(int(row["bbox_maxr"]) + pad, rgb.shape[0])
    maxc = min(int(row["bbox_maxc"]) + pad, rgb.shape[1])

    crop_rgb = rgb[minr:maxr, minc:maxc].copy()
    crop_mask = (labeled[minr:maxr, minc:maxc] == int(row["label_id"]))

    if crop_rgb.size == 0:
        return None

    # background putih untuk area luar object
    white_bg = np.ones_like(crop_rgb, dtype=np.uint8) * 255
    crop_object = white_bg.copy()
    crop_object[crop_mask] = crop_rgb[crop_mask]

    # outline object
    boundary = segmentation.find_boundaries(crop_mask, mode="outer")
    crop_object[boundary] = [255, 0, 0]

    h, w = crop_object.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(target_size / max(h, 1), target_size / max(w, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(crop_object, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas


def show_class_gallery(result_df, rgb, labeled, class_name, n_cols=6):
    class_df = result_df[result_df["predicted_class"] == class_name].copy()

    if class_df.empty:
        st.info(f"No objects in class: {class_name}")
        return

    st.markdown(f"### {class_name} ({len(class_df)})")

    rows = [class_df.iloc[i:i+n_cols] for i in range(0, len(class_df), n_cols)]

    for row_chunk in rows:
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            with cols[col_idx]:
                if col_idx < len(row_chunk):
                    row = row_chunk.iloc[col_idx]
                    thumb = crop_object_thumbnail(rgb, labeled, row, pad=8, target_size=96)
                    if thumb is not None:
                        st.image(thumb, use_container_width=True)
                        st.caption(
                            f"ID {int(row['label_id'])} | conf {row['confidence']:.2f}"
                        )


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.header("Controls")

    selected_class = st.selectbox("Active class", ANALYSIS_CLASSES)
    mark_as_excluded = st.checkbox("Mark clicked object as Excluded", value=False)
    show_unknown_gallery = st.checkbox("Show Excluded gallery", value=False)

    if st.button("Reset all annotations"):
        reset_all_annotations()
        st.rerun()

    st.markdown("---")
    st.write("Samples per analysis class")
    for cls in ANALYSIS_CLASSES:
        st.write(f"**{cls}**: {len(st.session_state.samples.get(cls, []))}")

    st.write(f"**Excluded objects**: {len(st.session_state.samples.get('Unknown', []))}")

# =========================================================
# IMAGE UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "Upload histopathology image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded is not None:
    if st.session_state.last_uploaded_name != uploaded.name:
        st.session_state.last_uploaded_name = uploaded.name
        reset_all_annotations()
        st.session_state.objects_df = None
        st.session_state.labeled_mask = None
        st.session_state.preview_rgb = None

    pil_img = Image.open(uploaded)
    rgb = pil_to_rgb_array(pil_img)

    gray, binary_mask, labeled = preprocess_and_segment(rgb)
    objects_df = extract_object_features(rgb, gray, labeled)

    st.session_state.objects_df = objects_df
    st.session_state.labeled_mask = labeled
    st.session_state.preview_rgb = rgb

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Image annotation")

        preview_for_click = annotate_image(
            rgb,
            objects_df,
            result_df=st.session_state.result_df,
            sample_dict=st.session_state.samples
        )

        display_img, scale = make_display_image(preview_for_click, max_width=800)
        click = streamlit_image_coordinates(display_img)

        if click is not None:
            real_x = int(click["x"] / scale)
            real_y = int(click["y"] / scale)

            nearest_id = find_nearest_object(
                real_x, real_y,
                st.session_state.objects_df,
                max_dist=35
            )

            if nearest_id is not None:
                target_class = "Unknown" if mark_as_excluded else selected_class

                if nearest_id not in st.session_state.samples[target_class]:
                    already_used = any(
                        nearest_id in st.session_state.samples[c]
                        for c in ALL_CLASSES
                    )
                    if not already_used:
                        st.session_state.samples[target_class].append(nearest_id)
                        st.rerun()

        st.caption("Click near a cell to assign it to the selected class, or mark it as Excluded.")

        c1, c2, c3 = st.columns(3)

        with c1:
            undo_target = "Unknown" if mark_as_excluded else selected_class
            if st.button(f"Undo last {'Excluded' if mark_as_excluded else selected_class}"):
                if len(st.session_state.samples[undo_target]) > 0:
                    st.session_state.samples[undo_target].pop()
                    st.rerun()

        with c2:
            clear_target = "Unknown" if mark_as_excluded else selected_class
            if st.button(f"Clear {'Excluded' if mark_as_excluded else selected_class}"):
                st.session_state.samples[clear_target] = []
                st.rerun()

        with c3:
            if st.button("Show segmentation preview"):
                seg_vis = make_colored_segmentation(
                    rgb=st.session_state.preview_rgb,
                    labeled=st.session_state.labeled_mask,
                    result_df=st.session_state.result_df,
                    sample_dict=st.session_state.samples,
                    objects_df=st.session_state.objects_df
                )
                st.image(seg_vis, caption="Colored segmentation", use_container_width=True)

    with right:
        st.subheader("Training and results")

        training_df = build_training_table(
            st.session_state.objects_df,
            st.session_state.samples
        )

        st.write("Training objects selected:", len(training_df))

        min_ok = sum(len(st.session_state.samples.get(cls, [])) >= 3 for cls in ANALYSIS_CLASSES)
        if min_ok < 2:
            st.warning("Pick at least 3 samples in at least 2 analysis classes for a usable first model.")

        if not training_df.empty:
            st.dataframe(
                training_df[["label_id", "target_class"] + feature_columns()].head(20),
                use_container_width=True
            )

        if st.button("Train and classify"):
            if training_df.empty:
                st.error("No training samples yet.")
            else:
                X_train = training_df[feature_columns()].copy()
                y_train = training_df["target_class"].copy()

                X_all = st.session_state.objects_df[feature_columns()].copy()

                X_train = X_train.fillna(X_train.median(numeric_only=True))
                X_all = X_all.fillna(X_train.median(numeric_only=True))

                clf = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                clf.fit(X_train, y_train)

                pred = clf.predict(X_all)
                proba = clf.predict_proba(X_all)
                max_proba = proba.max(axis=1)

                result_df = st.session_state.objects_df.copy()
                result_df["predicted_class"] = pred
                result_df["confidence"] = max_proba

                excluded_ids = set(st.session_state.samples.get("Unknown", []))
                if len(excluded_ids) > 0:
                    result_df.loc[result_df["label_id"].isin(excluded_ids), "predicted_class"] = "Unknown"
                    result_df.loc[result_df["label_id"].isin(excluded_ids), "confidence"] = 1.0

                st.session_state.result_df = result_df
                st.session_state.trained = True

        if st.session_state.trained and st.session_state.result_df is not None:
            result_df = st.session_state.result_df.copy()

            st.success("Classification complete")

            display_df = result_df[result_df["predicted_class"] != "Unknown"].copy()

            if display_df.empty:
                st.warning("All detected objects are currently excluded or classified as Unknown.")
            else:
                count_df = (
                    display_df["predicted_class"]
                    .value_counts()
                    .rename_axis("Class")
                    .reset_index(name="Count")
                )
                count_df["Percent"] = 100 * count_df["Count"] / count_df["Count"].sum()

                st.dataframe(count_df, use_container_width=True)

                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                ax_bar.bar(count_df["Class"], count_df["Count"])
                ax_bar.set_title("Cell Counts by Class")
                ax_bar.set_ylabel("Count")
                plt.xticks(rotation=30, ha="right")
                st.pyplot(fig_bar)
                plt.close(fig_bar)

                fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
                ax_pie.pie(
                    count_df["Count"],
                    labels=count_df["Class"],
                    autopct="%1.1f%%",
                    startangle=90
                )
                ax_pie.set_title("Cell Composition")
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            annotated = annotate_image(
                st.session_state.preview_rgb,
                st.session_state.objects_df,
                result_df=result_df,
                sample_dict=st.session_state.samples
            )
            st.image(annotated, caption="Annotated image", use_container_width=True)

            seg_result = make_colored_segmentation(
                rgb=st.session_state.preview_rgb,
                labeled=st.session_state.labeled_mask,
                result_df=result_df,
                sample_dict=st.session_state.samples,
                objects_df=st.session_state.objects_df
            )
            st.image(seg_result, caption="Class-colored segmentation", use_container_width=True)

            st.markdown("---")
            st.subheader("Object gallery by class")

            for cls in ANALYSIS_CLASSES:
                show_class_gallery(
                    result_df=display_df,
                    rgb=st.session_state.preview_rgb,
                    labeled=st.session_state.labeled_mask,
                    class_name=cls,
                    n_cols=6
                )

            if show_unknown_gallery:
                unknown_df = result_df[result_df["predicted_class"] == "Unknown"].copy()
                if not unknown_df.empty:
                    st.markdown("---")
                    st.subheader("Excluded object gallery")
                    show_class_gallery(
                        result_df=result_df,
                        rgb=st.session_state.preview_rgb,
                        labeled=st.session_state.labeled_mask,
                        class_name="Unknown",
                        n_cols=6
                    )

            export_df = result_df[result_df["predicted_class"] != "Unknown"].copy()
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv,
                file_name="histoinflam_results.csv",
                mime="text/csv"
            )

st.markdown("---")

st.markdown("""
<div style="
    text-align:left;
    color:#6b7280;
    font-size:13px;
    padding-top:10px;
    padding-bottom:10px;
    border-top:1px solid #e5e7eb;
    margin-top:20px;
">
© 2026 Mawar Subangkit<br>
<b>Automatic Annotated Cell Counter Software</b><br><br>

If you use this software, please cite:<br>

<b>Subangkit</b>, MAWAR (2026)<br>
<i>iRATco Cell Counter: Automatic Annotated Cell Counter Software</i><br>

<a href="available at: https://iratco-cell.streamlit.app/" target="_blank" style="color:#6b7280;">
available at: https://iratco-cell.streamlit.app/
</a>
</div>
""", unsafe_allow_html=True)
