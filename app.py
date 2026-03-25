import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

st.title("iRATco TrackR")

st.markdown("---")

st.markdown(
"""
© 2026 Mawar Subangkit  
Mouse Tracking Analysis Software  

If you use this software, please cite:

Subangkit M. (2026).  
**IRATCO TrackR: Open-field behavioral tracking software.**  
Available at: https://github.com/username/repository
"""
)

uploaded_video = st.file_uploader("Upload mouse video")


# -------------------------------------------------
# Mouse detection
# -------------------------------------------------

def detect_mouse(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _,mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

    coords = np.column_stack(np.where(mask>0))

    if len(coords)==0:
        return None,None

    y,x = coords.mean(axis=0)

    return int(x),int(y)


# -------------------------------------------------
# Run analysis
# -------------------------------------------------

if uploaded_video:

    if st.button("Run analysis"):

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        X=[]
        Y=[]

        progress = st.progress(0)

        frame_window = st.empty()

        col1,col2,col3 = st.columns(3)

        traj_plot = col1.empty()
        dist_plot = col2.empty()
        vel_plot = col3.empty()

        heat_plot = st.empty()

        frame_id = 0

        while True:

            ret,frame = cap.read()

            if not ret:
                break

            x,y = detect_mouse(frame)

            X.append(x)
            Y.append(y)

            if x is not None:
                cv2.circle(frame,(x,y),6,(0,0,255),-1)

            frame_window.image(frame,channels="BGR")

            track = pd.DataFrame({"X":X,"Y":Y})

            # -------------------------------------------------
            # Mirror vertical coordinate
            # -------------------------------------------------

            track["Y"] = height - track["Y"]

            # -------------------------------------------------
            # Smoothing trajectory
            # -------------------------------------------------

            track["Xs"] = track["X"].rolling(5,center=True).mean()
            track["Ys"] = track["Y"].rolling(5,center=True).mean()

            track["Xs"].fillna(track["X"],inplace=True)
            track["Ys"].fillna(track["Y"],inplace=True)

            if len(track)>2:

                track["dx"] = track.Xs.diff()
                track["dy"] = track.Ys.diff()

                track["step_distance"] = np.sqrt(track.dx**2 + track.dy**2)

                track["velocity"] = track["step_distance"]

                track["cumulative_distance"] = track.step_distance.fillna(0).cumsum()

                track["bearing"] = np.arctan2(track.dy, track.dx)

                track["bearing_deg"] = np.degrees(track["bearing"])

                track["turn_angle"] = track["bearing_deg"].diff()

                track["turn_angle"] = (track["turn_angle"] + 180) % 360 - 180


                # update plot tiap 10 frame agar cepat
                if frame_id % 10 == 0:

                    # ----------------------------------
                    # Trajectory
                    # ----------------------------------

                    fig1,ax1 = plt.subplots()

                    ax1.plot(track.Xs,track.Ys,color="red")

                    ax1.set_aspect("equal")

                    ax1.set_title("Trajectory")

                    traj_plot.pyplot(fig1)

                    plt.close(fig1)


                    # ----------------------------------
                    # Cumulative distance
                    # ----------------------------------

                    fig2,ax2 = plt.subplots()

                    ax2.plot(track["cumulative_distance"],color="steelblue")

                    ax2.set_title("Cumulative distance")

                    dist_plot.pyplot(fig2)

                    plt.close(fig2)


                    # ----------------------------------
                    # Velocity
                    # ----------------------------------

                    fig3,ax3 = plt.subplots()

                    ax3.plot(track["velocity"],color="purple")

                    ax3.set_title("Velocity")

                    vel_plot.pyplot(fig3)

                    plt.close(fig3)


                    # ----------------------------------
                    # Heatmap
                    # ----------------------------------

                    if len(track)>20:

                        fig4,ax4 = plt.subplots()

                        sns.kdeplot(
                            x=track.Xs,
                            y=track.Ys,
                            fill=True,
                            cmap="RdYlGn_r",
                            ax=ax4
                        )

                        ax4.set_aspect("equal")

                        heat_plot.pyplot(fig4)

                        plt.close(fig4)

            frame_id += 1

            progress.progress(frame_id/total_frames)

        cap.release()

        st.success("Analysis complete")


        # -------------------------------------------------
        # Directional analysis
        # -------------------------------------------------

        st.subheader("Directional analysis")

        col4,col5 = st.columns(2)

        bins = np.linspace(-180,180,24)

        # Absolute bearing

        with col4:

            fig5 = plt.figure(figsize=(4,4))

            hist,_ = np.histogram(track["bearing_deg"].dropna(), bins=bins)

            theta = np.deg2rad((bins[:-1]+bins[1:])/2)

            ax5 = fig5.add_subplot(111, polar=True)

            ax5.bar(theta,hist,width=np.deg2rad(15),
                    color="steelblue")

            ax5.set_title("Absolute bearing")

            st.pyplot(fig5)


        # Turn direction

        with col5:

            fig6 = plt.figure(figsize=(4,4))

            hist,_ = np.histogram(track["turn_angle"].dropna(), bins=bins)

            theta = np.deg2rad((bins[:-1]+bins[1:])/2)

            ax6 = fig6.add_subplot(111, polar=True)

            ax6.bar(theta,hist,width=np.deg2rad(15),
                    color="tomato")

            ax6.set_title("Turn direction")

            st.pyplot(fig6)


        # -------------------------------------------------
        # Download CSV
        # -------------------------------------------------

        csv = track.to_csv(index=False)

        st.download_button(
            "Download tracking data",
            csv,
            "tracking.csv"
        )

        