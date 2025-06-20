# **Diffusion Visualization Explorer - Web**

## **About The Project**

**Web DiVE** is a lightweight, browser-based visualization and analysis tool for tractography data. As a streamlined extension of the [DiVE](https://github.com/USC-LoBeS/dive) CLI, it enables users to explore and analyze streamline bundles directly within a web browser—no installation required. Web DiVE is especially useful for quick data inspection, collaborative presentations, and interactive demos.

The application is organized into two main components:

- **Tracts View** – A 3D scene for visualizing white matter streamlines. Users can modify bundle properties, perform statistical analysis, and generate PNG snapshots of the scene.
- **Table View** – A panel for uploading and visualizing statistical `.csv` data. This data can be overlaid on tract bundles using a range of coloring and mapping options.

![Web Dive UI](/images/ui.png)

### Key Features

- **Upload and Visualize Tractography Files**: Supports `.trk` files for displaying white matter bundles in an interactive 3D scene.
- **Interactive Scene Controls**: Includes camera tools, anatomical orientation presets, and overlay options such as Glass Brain and anatomical slice views.
- **Color Customization**: Streamlines can be colored by bundle, anatomical direction (RGB orientation), or overlaid with statistical metrics using matplotlib colormaps.
- **Statistical Table Integration**: Overlay scalar metrics from `.csv` files onto bundles with dynamic mapping controls.
- **Lightweight and Real-Time**: Optimized for browser performance, assuming streamline alignment to MNI space. For subject-space or advanced analyses, use the full DiVE CLI.
- **PNG Export**: Capture and save visual snapshots of the current scene with a single click.

### Segmentation and Statistical Overlays

Web DiVE provides two built-in segmentation tools for bundle-level analysis:

- **Center Line Segmentation**: Uses QuickBundles clustering to identify a representative core fiber for bundle segmentation, enabling group-wise comparisons.
- **Medial Tractography Analysis (MeTA)**: Applies dynamic time warping to extract anatomically consistent and topologically aligned core bundles across subjects.

After segmentation, users can apply statistical overlays using pre-defined matplotlib colormaps, selectable from the color wheel interface.

---

## **Live Demo**

Try out the application directly in your browser:  
[**Diffusion Visualization Explorer - Web**](https://brainescience.shinyapps.io/dive/)

---

## **Usage**

We have included example `.trk` files in this repository, which can be used to test the application's functionality.

- These test files are located in the `/data/bundles/` folder.
- Additionally, a `stat_template.csv` file is provided for segmentation of uploaded bundles.

### **To Get Started**

Clone the repository to access the data:

```bash
git clone https://github.com/USC-LoBeS/Web-Dive.git
cd web-dive
```

## Contact

Lobes - [Aarya Vakharia](mailto:avakhari@usc.edu), [Neda Janahshad](mailto:njahansh@usc.edu)

## Acknowledgements

Web DiVE was developed as part of the DiVE (Diffusion MRI Visualization & Exploration) ecosystem. We gratefully acknowledge:

- [Siddharth Narula, Iyad Ba Gari, Aarya Vakharia, Neda Jahanshad, "Web DiVE - Tractography Anywhere. Organization for Human Brain Mapping (OHBM 2025) June 24,2025](https://drive.google.com/file/d/1Kl4Mh6r5AdBG24PGWuRKiNWjlSDoHnH8/view)

- Open-source libraries including [Three.js](https://threejs.org/), [Plotly](https://plotly.com/javascript/), and [Matplotlib](https://matplotlib.org/) for enabling browser-based scientific visualization.

- [Siddharth Narula, Iyad Ba Gari, Shruti P. Gadewar, Sunanda Somu, Neda Jahanshad, "Diffusion Visualization Explorer (DiVE) Organization for Human Brain Mapping (OHBM 2024) June 26,2024](https://drive.google.com/file/d/1dsYLTrbfHmrlJNzih-CqbMye32q3sPfU/view)

- [Iyad Ba Gari, Shayan Javid, Alyssa H. Zhu, Shruti P. Gadewar, Siddharth Narula, Abhinaav Ramesh, Sophia I. Thomopoulos et al. "Along-Tract Parameterization of White Matter Microstructure using Medial Tractography Analysis (MeTA)." In 2023 19th International Symposium on Medical Information Processing and Analysis (SIPAIM), pp. 1-5. IEEE, 2023.](https://doi.org/10.1109/SIPAIM56729.2023.10373540)
