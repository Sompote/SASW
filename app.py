import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert

def generate_sasw_data(num_points=1000, duration=1.0):
    time = np.linspace(0, duration, num_points)
    frequency = 50  # Hz
    amplitude = np.sin(2 * np.pi * frequency * time) * np.exp(-5 * time)
    noise = np.random.normal(0, 0.1, num_points)
    return time, amplitude + noise

def save_to_csv(time, amplitude, filename):
    df = pd.DataFrame({'time': time, 'amplitude': amplitude})
    df.to_csv(filename, index=False, header=False)

def plot_raw_data(time, amplitude):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, amplitude)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Raw SASW Data")
    return fig

def calculate_phase_velocity(time, amplitude, d):
    frequencies = np.fft.fftfreq(len(time), time[1] - time[0])
    fft = np.fft.fft(amplitude)
    positive_freq_mask = frequencies > 0
    frequencies = frequencies[positive_freq_mask]
    fft = fft[positive_freq_mask]
    
    phase = np.angle(fft)
    unwrapped_phase = np.unwrap(phase)
    
    velocity = 2 * np.pi * frequencies * d / unwrapped_phase
    velocity = np.abs(velocity)  # Take absolute value to avoid negative velocities
    
    return frequencies, velocity

def power_law(x, a, b):
    return a * x**b

def invert_dispersion_curve(wavelengths, phase_velocities):
    initial_guess = [1.0, 0.5]
    popt, _ = curve_fit(power_law, wavelengths, phase_velocities, p0=initial_guess, maxfev=5000)

    a, b = popt
    depths = np.linspace(min(wavelengths)/3, max(wavelengths)/3, 100)
    shear_velocities = power_law(depths, a, b)

    return depths, shear_velocities

def calculate_layer_velocities(depths, shear_velocities, num_layers):
    layer_depths = np.linspace(min(depths), max(depths), num_layers + 1)
    layer_velocities = []

    for i in range(num_layers):
        mask = (depths >= layer_depths[i]) & (depths < layer_depths[i+1])
        layer_velocities.append(np.mean(shear_velocities[mask]))

    return layer_depths[1:], layer_velocities

def plot_layered_profile(layer_depths, layer_velocities):
    fig, ax = plt.subplots()
    for i in range(len(layer_depths)):
        ax.plot([layer_velocities[i], layer_velocities[i]], [layer_depths[i-1] if i > 0 else 0, layer_depths[i]], 'b-')
    ax.set_xlabel("Shear Wave Velocity (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Layered Shear Wave Velocity Profile")
    ax.invert_yaxis()
    return fig

def main():
    st.title("SASW Analysis App")
    
    if st.button("Generate Example Data"):
        time, amplitude = generate_sasw_data()
        save_to_csv(time, amplitude, "example_sasw_data.csv")
        st.success("Example data generated and saved as 'example_sasw_data.csv'")
        
        # Automatically load the generated data
        data = pd.DataFrame({'time': time, 'amplitude': amplitude})
        st.session_state.data = data
        st.session_state.file_name = "example_sasw_data.csv"

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, header=None, names=['time', 'amplitude'])
        st.session_state.data = data
        st.session_state.file_name = uploaded_file.name
    
    if 'data' in st.session_state:
        st.subheader(f"Raw Data ({st.session_state.file_name})")
        raw_data_fig = plot_raw_data(st.session_state.data['time'], st.session_state.data['amplitude'])
        st.pyplot(raw_data_fig)
        
        d = st.number_input("Enter the distance between geophones (m)", min_value=0.1, value=1.0)
        num_layers = st.number_input("Enter the number of layers for inversion", min_value=1, max_value=10, value=4, step=1)
        
        if st.button("Calculate Dispersion Curve and Vs Profile"):
            time = st.session_state.data['time']
            amplitude = st.session_state.data['amplitude']
            
            # Calculate dispersion curve
            frequencies, phase_velocities = calculate_phase_velocity(time, amplitude, d)
            wavelengths = 1 / frequencies
            
            # Plot dispersion curve
            st.subheader("Dispersion Curve")
            fig, ax = plt.subplots()
            ax.scatter(wavelengths, phase_velocities, alpha=0.5)
            ax.set_xlabel("Wavelength (m)")
            ax.set_ylabel("Phase Velocity (m/s)")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title("Experimental Dispersion Curve")
            st.pyplot(fig)
            
            # Invert dispersion curve
            depths, shear_velocities = invert_dispersion_curve(wavelengths, phase_velocities)
            
            if depths is not None and shear_velocities is not None:
                # Plot continuous Vs profile
                st.subheader("Continuous Shear Wave Velocity Profile")
                fig, ax = plt.subplots()
                ax.plot(shear_velocities, depths)
                ax.set_xlabel("Shear Wave Velocity (m/s)")
                ax.set_ylabel("Depth (m)")
                ax.set_title("Continuous Vs Profile")
                ax.invert_yaxis()
                st.pyplot(fig)

                # Calculate and plot layered profile
                layer_depths, layer_velocities = calculate_layer_velocities(depths, shear_velocities, num_layers)
                
                st.subheader("Layered Shear Wave Velocity Profile")
                layered_fig = plot_layered_profile(layer_depths, layer_velocities)
                st.pyplot(layered_fig)

                st.subheader("Layer Velocities")
                for i, (depth, velocity) in enumerate(zip(layer_depths, layer_velocities)):
                    st.write(f"Layer {i+1}: Depth = {depth:.2f} m, Vs = {velocity:.2f} m/s")
            else:
                st.error("Failed to invert dispersion curve. Please check your input data.")

if __name__ == "__main__":
    main()