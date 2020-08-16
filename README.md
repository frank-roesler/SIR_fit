# SIR_fit
Generates simulated data and fits a infectious disease (SIR) model.

This Python app provides a GUI that allows the user to generate simulated data points for an infectious disease wave and then fit a simple [SIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model_without_vital_dynamics) to that data.
The user can choose between steepest descent or conjugate gradient methods. Experimentally, the conjugate gradient method is considerably faster than the steepest descent method, but is also more unstable and does not always find the minimum.  
To run the application, download the two .py files, navigate to to their location in the terminal and type "pythonw GUI.py".
