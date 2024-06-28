import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.stats import chi2

r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6,1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3,1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)
    
    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)

    def update_step(self):
        Inn = self.Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z/np.sqrt(x**2 + y**2)) * 180/3.14
    az = math.atan(y/x)    
    if x > 0.0:                
        az = 3.14/2 - az
    else:
        az = 3*3.14/2 - az       
    az = az * 180 / 3.14 
    if az < 0.0:
        az = 360 + az
    if az > 360:
        az = az - 360      
    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i]/np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i]/x[i]))    
        if x[i] > 0.0:                
            az[i] = 3.14/2 - az[i]
        else:
            az[i] = 3*3.14/2 - az[i]       
        az[i] = az[i] * 180 / 3.14 
        if az[i] < 0.0:
            az[i] = 360 + az[i]
        if az[i] > 360:
            az[i] = az[i] - 360
    return r, az, el

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)
            print("Cartesian coordinates (x, y, z):", x, y, z)
            r, az, el = cart2sph(x, y, z)
            print("Spherical coordinates (r, az, el):", r, az, el)
            measurements.append((r, az, el, mt))
    return measurements

def chi_square_test(track, measurement, threshold):
    # Compute the innovation (difference between predicted and actual measurement)
    innovation = measurement - np.dot(track.H, track.Sf)
    S = np.dot(track.H, np.dot(track.Pf, track.H.T)) + track.R
    chi_square_value = np.dot(innovation.T, np.dot(np.linalg.inv(S), innovation))
    return np.all(chi_square_value <= threshold)


def group_measurements_by_time(measurements):
    measurement_groups = {}
    for meas in measurements:
        time = meas[3]
        if time not in measurement_groups:
            measurement_groups[time] = []
        measurement_groups[time].append(meas)
    return measurement_groups

def calculate_joint_probability(track, meas):
    # Calculate the likelihood of the measurement given the track's state
    innovation = meas[:3] - np.dot(track['filter'].H, track['filter'].Sf)
    S = np.dot(track['filter'].H, np.dot(track['filter'].Pf, track['filter'].H.T)) + track['filter'].R
    likelihood = np.exp(-0.5 * np.dot(innovation.T, np.dot(np.linalg.inv(S), innovation)))
    return likelihood

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Read measurements from CSV file
csv_file_path = 'ttk_50.csv'
measurements = read_measurements_from_csv(csv_file_path)

# Group measurements by time
measurement_groups = group_measurements_by_time(measurements)

# Track initiation
tracks = []
track_id = 0
for time, group in measurement_groups.items():
    if track_id == 0:
        # Initialize tracks with the first group
        for meas in group:
            r, az, el, mt = meas
            kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
            tracks.append({'id': track_id, 'filter': kalman_filter, 'measurements': [meas]})
            track_id += 1
    else:
        for meas in group:
            assigned = False
            for track in tracks:
                if chi_square_test(track['filter'], meas[:3], threshold=0.95):
                    track['measurements'].append(meas)
                    assigned = True
                    break
            if not assigned:
                kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
                tracks.append({'id': track_id, 'filter': kalman_filter, 'measurements': [meas]})
                track_id += 1

csv_file_predicted = "ttk_50.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['FT', 'FX', 'FY', 'FZ']].values

A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)
number = 1000
result = np.divide(A[0], number)

time_list = []
r_list = []
az_list = []
el_list = []

for track in tracks:
    for meas in track['measurements']:
        mt = meas[3]
        r, az, el = meas[:3]
        time_list.append(mt + 0.013)
        r_list.append(r)
        az_list.append(az)
        el_list.append(el)

    max_prob_meas = None
    max_prob = 0
    for meas in track['measurements']:
        prob = calculate_joint_probability(track, meas)
        if prob > max_prob:
            max_prob = prob
            max_prob_meas = meas

    if max_prob_meas:
        track['filter'].Z = np.array(max_prob_meas[:3]).reshape((3, 1))
        track['filter'].update_step()

plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Range (r)', color='black')
plt.title('Range vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Azimuth (az)', color='black')
plt.title('Azimuth vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Elevation (el)', color='black')
plt.title('Elevation vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()
