# from scipy.spatial import ConvexHul
from os.path import join, dirname, basename
import scipy.stats as stats
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'


## Calculate all SPT properties posible (based on classical diffusion models) and store them together with the raw data in an AIO (all in one) format. One AIO file is saved for each individual input file.
## By default, the tracks are already >= 5 frames, enough for the 3-points MSD-tau fitting.

# scalling factors for physical units
um_per_pixel = 0.117
print("Please enter the s per frame for the video")
s_per_frame = float(input())
print(
    "Scaling factors: s_per_frame = "
    + str(s_per_frame)
    + ", um_per_pixel = "
    + str(um_per_pixel)
)


print("Choose the track csv files for processing:")
lst_fpath = list(fd.askopenfilenames())


# Output file columns
angle_bins = np.linspace(0, 180, 7).astype(int)  # #boundaries = #bins + 1
lst_angle_fraction_titles = [
    "(" + str(angle_bins[i]) + "," + str(angle_bins[i + 1]) + "]"
    for i in range(len(angle_bins) - 1)
]
columns = [
    "trackID",
    "list_of_t",
    "list_of_x",
    "list_of_y",
    "N_steps",
    "displacement_nm",
    "mean_stepsize_nm",
    "max_d_anytwo_nm",
    "mean_x_pxl",
    "mean_y_pxl",
    "mean_spot_intensity_max_in_track",
    "list_of_MSD_um2",
    "list_of_tau_s",
    "linear_fit_slope",
    "linear_fit_R2",
    "linear_fit_sigma",
    "linear_fit_D_um2s",
    "linear_fit_log10D",
    "loglog_fit_R2",
    "loglog_fit_log10D",
    "alpha",
    "list_of_angles",
] + lst_angle_fraction_titles


def calc_MSD_NonPhysUnit(df_track, lags):
    df_track_sorted = df_track.sort_values("t")
    # prepare storage arrays
    MSDs = []
    # STDs = np.array([], dtype=float)

    # filling gaps
    ref_t = list(range(df_track_sorted.t.min(), df_track_sorted.t.max()))
    missing_t = list(set(ref_t) - set(df_track_sorted.t))
    if len(missing_t) > 0:
        for t in missing_t:
            complete_t = np.append(df_track_sorted.t, np.array(missing_t, dtype=int))
            complete_x = np.append(df_track_sorted.x, np.repeat(np.NaN, len(missing_t)))
            complete_y = np.append(df_track_sorted.y, np.repeat(np.NaN, len(missing_t)))
            df_complete = pd.DataFrame()
            df_complete["x"] = complete_x
            df_complete["y"] = complete_y
            df_complete["t"] = complete_t
            df_complete = df_complete.sort_values("t")

    else:
        df_complete = df_track_sorted

    # calculate MSDs corresponding to a series of lag times
    for lag in lags:
        Xs = np.array(df_complete.x, dtype=float)
        Ys = np.array(df_complete.y, dtype=float)

        SquareDisplacements = (Xs[lag:] - Xs[:-lag]) ** 2 + (Ys[lag:] - Ys[:-lag]) ** 2
        MSD = np.nanmean(SquareDisplacements)
        MSDs.append(MSD)

    MSDs = np.array(MSDs, dtype=float)
    return MSDs


def calc_angle(x, y):
    # x and y at time 0 and time 1
    x0 = x[:-1]
    x1 = x[1:]
    y0 = y[:-1]
    y1 = y[1:]
    # unit vectors of all steps, and step 0 and step 1
    vector = np.array([x1 - x0, y1 - y0])
    # convert to complex number to use np.angle
    vector_complex = 1j * vector[1, :]
    vector_complex += vector[0, :]
    angles_eachstep = np.angle(vector_complex, deg=True)
    angles = np.ediff1d(angles_eachstep)  # between adjacent steps
    # convert all angles to within range (0,+-180) for output
    angles[angles < -180] = angles[angles < -180] + 360
    angles[angles > 180] = angles[angles > 180] - 360

    return angles


print("Now Processing:", dirname(lst_fpath[0]))
for fpath in track(lst_fpath):
    df_current_file = pd.read_csv(fpath, dtype=float)
    df_current_file = df_current_file.astype({"t": int})
    fname = basename(fpath)
    lst_trackID_in_file = df_current_file.trackID.unique().tolist()

    lst_rows_of_df = []
    for trackID in lst_trackID_in_file:
        df_current_track = df_current_file[df_current_file.trackID == trackID]
        tracklength = df_current_track.shape[0]

        # filter out short tracks, so spots_reformatted.csv can be used as tracks
        if tracklength < 5:
            continue

        x = df_current_track["x"].to_numpy(dtype=float)
        y = df_current_track["y"].to_numpy(dtype=float)
        disp_nm = (
            np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2) * um_per_pixel * 1000
        )
        mean_stepsize_nm = (
            np.mean(np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))
            * um_per_pixel
            * 1000
        )
        lags = np.arange(1, tracklength)
        MSDs = calc_MSD_NonPhysUnit(df_current_track, lags)
        lags_phys = lags * s_per_frame
        MSDs_phys = MSDs * (um_per_pixel**2)  # um^2

        ## Remove artificial tracks from dead pixels, which will have fixed value x or y like [105.0, 106.0, 105.0, 105.0, 105.0, 105.0, 106.0]
        _, counts_x = np.unique(x, return_counts=True)
        _, counts_y = np.unique(y, return_counts=True)
        if counts_x.max() > 2 or counts_y.max() > 2:
            continue

        ## To determine if a molecuel is static or mobile, independent of Einstein's diffusion equation
        max_d_anytwo_nm = (
            np.max(
                np.sqrt((x - np.atleast_2d(x).T) ** 2 + (y - np.atleast_2d(y).T) ** 2)
            )
            * um_per_pixel
            * 1000
        )
        # equivalent diameter is calculated from the convex hull of all positions within a track, namely the space a molecule has surveyed.
        # points_coordinates_nm = np.stack([x, y], axis=-1) * um_per_pixel * 1000
        # convex_area_nm2 = ConvexHull(
        #     points_coordinates_nm
        # ).volume  # bug fix: ConvexHull.area is perimeter in 2d
        # equivalent_d_nm = np.sqrt(convex_area_nm2 / np.pi) * 2

        ## D formula with errors (MSD: um^2, t: s, D: um^2/s, n: dimension, R: motion blur coefficient; doi:10.1103/PhysRevE.85.061916)

        # From the paper's fig 7, the optimal number of fitting points p_min is almost always roughly half of total MSD points, with a minimum p_min = 3 (asuming error x > 1)
        if round(lags.shape[0] / 2) < 3:
            lags_to_fit = lags[:3]
        else:
            lags_to_fit = lags[: round(lags.shape[0] / 2)]

        # diffusion dimension = 2. Note: This is the dimension of the measured data, not the actual movements! Although particles are doing 3D diffussion, the microscopy data is a projection on 2D plane and thus should be treated as 2D diffusion!
        # MSD = 2 n D tau + 2 n sigma^2 - 4 n R D tau, n=2, R=1/6
        # Therefore, slope = (2n-4nR)D = (8/3) D; intercept = 2 n sigma^2
        slope_linear, intercept_linear, R_linear, P, std_err = stats.linregress(
            lags_to_fit * s_per_frame, MSDs_phys[: len(lags_to_fit)]
        )
        if slope_linear > 0:
            # Actually, if any of the slope or intercept is negative, the model does not technically fit. However, it seems faster molecules will more likely to ended up with very good fitting with a small negative intercept. It's not reasonable to exclude these molecules. Therefore, move the if statement for intercept to inside so these fast molecules are recorded.
            D_phys_linear = slope_linear / (8 / 3)  # um^2/s
            log10D_linear = np.log10(D_phys_linear)
            if intercept_linear >= 0:
                sigma_phys = np.sqrt(intercept_linear / 4) * 1000  # nm
            else:
                sigma_phys = np.NaN
        else:
            D_phys_linear = np.NaN
            log10D_linear = np.NaN
            sigma_phys = np.NaN

        # MSD = 2 n D tau^alpha = 4 D tau^alpha
        # log(MSD) = alpha * log(tau) + log(D) + log(4)
        # Therefore, slope = alpha; intercept = log(D) + log(4)
        slope_loglog, intercept_loglog, R_loglog, P, std_err = stats.linregress(
            np.log10(lags_to_fit * s_per_frame),
            np.log10(MSDs_phys[: len(lags_to_fit)]),
        )
        log10D_loglog = intercept_loglog - np.log10(4)
        alpha = slope_loglog

        angles = calc_angle(x, y)
        densities, _ = np.histogram(np.absolute(angles), angle_bins, density=True)
        # fractions are summed to 1; fraction = density * bin width
        fractions = densities * (angle_bins[1] - angle_bins[0])

        # Save
        new_row = [
            trackID,
            df_current_track["t"].to_list(),
            df_current_track["x"].to_list(),
            df_current_track["y"].to_list(),
            tracklength,
            disp_nm,
            mean_stepsize_nm,
            max_d_anytwo_nm,
            x.mean(),
            y.mean(),
            df_current_track["meanIntensity"].max(),  # max of mean spot intensity
            MSDs_phys.tolist(),
            lags_phys.tolist(),
            slope_linear,
            R_linear**2,
            sigma_phys,
            D_phys_linear,
            log10D_linear,
            R_loglog**2,
            log10D_loglog,
            alpha,
            angles.tolist(),
        ] + fractions.tolist()
        lst_rows_of_df.append(new_row)

    df_save = pd.DataFrame.from_records(
        lst_rows_of_df,
        columns=columns,
    )
    fname_save = join(dirname(fpath), "SPT_results_AIO-" + basename(fpath))
    df_save.to_csv(fname_save, index=False)
