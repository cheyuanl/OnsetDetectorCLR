### OnsetDetectorCLR
--

OnsetDetectorCLR is an onset detector based on Constrained Linear Reconstruction, which is a extension of spectral-flux-based onset detector.

For more details of this program, please refer to:<br>
*“Musical Onset Detection Using Constrained Linear Reconstruction”, IEEE Signal Process. Lett., Vol. 22, Issue: 11, 2015.*

####Dependencies

python 2.7, numpy1.92, scipy15.1 (Optional: matplotlib1.43, mir_eval (for figure plot))


####Usage

````
python onsetDetectorCLR.py input.wav <optional arguments>
````
The script will output the .txt file with estimated onset time instances in second.

use ```-h``` to check for available arguments

####Credit
This program includes SuperFlux. Software credit:
Copyright (c) 2012 - 2014 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.
