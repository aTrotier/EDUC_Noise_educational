Noise in MRI (2019) - Educational and practical guide
====================================================

This is the code repository of the CRMSB (CNRS - University of Bordeaux / UMR5536), Bordeaux, France

> How noise works in MRI


Contributors
------------

Several people contributed to this repository by code, feedback, review and sharing
insights of their experience:

- Aurélien TROTIER
- Jean-Michel FRANCONI
- Eric THIAUDIERE

External Code used :
None


Requirements
------------

This code is written in Matlab, 2016a, but most likely works with earlier or later versions.



Getting Started
---------------

### Matlab
1. Clone this repository into a project folder of your liking, e.g. via
   `git clone https://github.com/mrtm-zurich/rrsg-arbitrary-sense.git rrsg-arbitrary-sense`
   - Note: The `data` subfolder of the repo already contains the example data from
   http://www.user.gwdg.de/~muecker1/rrsg_challenge.zip
2. Run `code/main.m`.
    - The main script calls three functions creating the Figures 4 to 6 from the original paper.
	- They should correspond to figure 4,5, and 6 in the original MRM paper.
	- The created Matlab Figures are saved as PNGs to the `results/` subfolder of the project folder.
3. Note: If you just want to try out the reconstruction pipeline (load data, get SENSE map, run recon)
   for one undersampling factor), run `demoRecon.m` instead of `main.m` in the
   `code` subfolder of the repository.


Results
-------

Figures
-------

### Figure
![Figure](code/html/noise_script_05.png?raw=true "Figure")

