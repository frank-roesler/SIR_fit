import wx
import SIR_fit
import matplotlib.pyplot as pp
from matplotlib.gridspec import GridSpec
import numpy as np
import time

def timeit(method):
    """Decorator to time the evaluation of a function"""
    def timed(self, data):
        ts = time.time()
        result = method(self, data)
        te = time.time()
        print('Elapsed time:', te-ts)
        return result
    return timed

class SIR_GUI(wx.Frame):
    """Implements a GUI for the SIR_fit class that allows the user to
    generate simulated data and fit a model by clicking buttons."""
    def __init__(self):
        super().__init__(parent=None, title='SIR fit', size=(530,400), style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)

        self.S0 = 1        # Initial value for "Susceptible"
        self.I0 = 0.01     # Initial value for "Infected"
        self.R0 = 0        # Initial value for "Removed"
        self.N_times = 150  # Times at which measurements are taken
        self.beta0 = 0.4
        self.gamma0 = 0.1
        self.T = 100
        self.dt = 0.05

        self.SIR = SIR_fit.SIR_fit(self.dt, self.T, self.S0, self.I0, self.R0, self.N_times)
        self.data_generated = False
        self.dialog = None

        panel1 = wx.Panel(self)

        # Create the BoxSizer to use for the Frame
        vertical_box_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vertical_box_sizer)

        # Add the Panel to the Frames Sizer
        vertical_box_sizer.Add(panel1, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)
        grid1 = wx.GridBagSizer(11, 3)
        # Static text:
        welcome = wx.StaticText(panel1, label='Please enter parameter values:')
        font = wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        welcome.SetFont(font)
        fit_method_text = wx.StaticText(panel1, label='Select fit method:')
        current_values = wx.StaticText(panel1, label='Current values:')
        font = wx.Font(13, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        current_values.SetFont(font)

        # Entered values:
        self.S_text = wx.StaticText(panel1, label='S0 = '+str(self.S0))
        self.I_text = wx.StaticText(panel1, label='I0 = '+str(self.I0))
        self.R_text = wx.StaticText(panel1, label='R0 = '+str(self.R0))
        self.b_text = wx.StaticText(panel1, label='beta0 = '+str(self.beta0))
        self.g_text = wx.StaticText(panel1, label='gamma0 = '+str(self.gamma0))
        self.N_text = wx.StaticText(panel1, label='N = ' + str(self.N_times))

        # Text fields:
        S_label      = wx.StaticText(panel1,label='"Susceptibles" initial value:')
        self.S_field = wx.TextCtrl(panel1, style=wx.TE_PROCESS_ENTER, size=(-1,-1))

        I_label = wx.StaticText(panel1, label='"Infected" initial value:')
        self.I_field = wx.TextCtrl(panel1, style=wx.TE_PROCESS_ENTER, size=(-1, -1))

        R_label = wx.StaticText(panel1, label='"Removed" initial value:')
        self.R_field = wx.TextCtrl(panel1, style=wx.TE_PROCESS_ENTER, size=(-1, -1))

        b_label = wx.StaticText(panel1, label='Actual value for beta:')
        self.b_field = wx.TextCtrl(panel1, style=wx.TE_PROCESS_ENTER, size=(-1, -1))
        self.b_field.SetHint('0<beta<1')

        g_label = wx.StaticText(panel1, label='Actual value for gamma:')
        self.g_field = wx.TextCtrl(panel1, style=wx.TE_PROCESS_ENTER, size=(-1, -1))
        self.g_field.SetHint('0<gamma<1')

        N_label = wx.StaticText(panel1, label='Number of data points:')
        self.N_field = wx.TextCtrl(panel1, style=wx.TE_PROCESS_ENTER, size=(-1, -1))

        # Buttons:
        enter_button = wx.Button(panel1, label='Enter values')
        enter_button.Bind(wx.EVT_BUTTON, self.vars_setter)

        simulate_button = wx.Button(panel1, label='Show data')
        simulate_button.Bind(wx.EVT_BUTTON, self.plot_simulated_data)

        self.dropdown = wx.Choice(panel1, choices=['Steepest descent', 'Conjugate gradient'])

        fit_button = wx.Button(panel1, label='Fit')
        fit_button.Bind(wx.EVT_BUTTON, self.fit_selection)

        reset_button = wx.Button(panel1, label='Reset')
        reset_button.Bind(wx.EVT_BUTTON, self.reset_app)

        # Lines:
        line = wx.StaticLine(panel1)

        # Add everything to the sizer:
        grid1.Add(welcome, pos=(0, 0))

        grid1.Add(current_values, pos=(0,2))

        grid1.Add(S_label, pos=(1,0))
        grid1.Add(self.S_field, pos=(1, 1))
        grid1.Add(self.S_text, pos=(1, 2))

        grid1.Add(I_label, pos=(2, 0))
        grid1.Add(self.I_field, pos=(2, 1))
        grid1.Add(self.I_text, pos=(2, 2))

        grid1.Add(R_label, pos=(3, 0))
        grid1.Add(self.R_field, pos=(3, 1))
        grid1.Add(self.R_text, pos=(3, 2))

        grid1.Add(b_label, pos=(4, 0))
        grid1.Add(self.b_field, pos=(4, 1))
        grid1.Add(self.b_text, pos=(4, 2))

        grid1.Add(g_label, pos=(5, 0))
        grid1.Add(self.g_field, pos=(5, 1))
        grid1.Add(self.g_text, pos=(5, 2))

        grid1.Add(N_label, pos=(6, 0))
        grid1.Add(self.N_field, pos=(6, 1))
        grid1.Add(self.N_text, pos=(6, 2))

        grid1.Add(enter_button, pos=(7, 0))

        grid1.Add(line, pos=(8, 0), span=(0, 3),
                  flag=wx.EXPAND | wx.BOTTOM, border=1)

        grid1.Add(fit_method_text, pos=(9, 1))
        grid1.Add(self.dropdown, pos=(9, 2))
        grid1.Add(simulate_button, pos=(9, 0))
        grid1.Add(fit_button, pos=(10, 1))
        grid1.Add(reset_button, pos=(10, 0))

        for i in range(10):
            grid1.AddGrowableRow(i)
            if i <3:
                grid1.AddGrowableCol(i)

        panel1.SetSizer(grid1)

    def vars_setter(self,evt):
        def set_vars(field, text, label):
            try:
                # Get values entered in text fields:
                if field==self.N_field:
                    entered = int(field.GetLineText(0))
                else:
                    entered = float(field.GetLineText(0))
                field.Clear()
                text.SetLabel(label + str(entered))
                self.data_generated = False
                self.dialog=None
                return entered
            except:
                self.dialog = wx.MessageDialog(None, message="All fields must have numerical values entered.",
                                          style=wx.OK)

        if self.S_field.GetLineText(0) != '':
            self.S0 = set_vars(self.S_field, self.S_text, 'S0 = ')
        if self.I_field.GetLineText(0) != '':
            self.I0 = set_vars(self.I_field, self.I_text, 'I0 = ')
        if self.R_field.GetLineText(0) != '':
            self.R0 = set_vars(self.R_field, self.R_text, 'R0 = ')
        if self.b_field.GetLineText(0) != '':
            self.beta0 = set_vars(self.b_field, self.b_text, 'beta0 = ')
        if self.g_field.GetLineText(0) != '':
            self.gamma0 = set_vars(self.g_field, self.g_text, 'gamma0 = ')
        if self.N_field.GetLineText(0) != '':
            self.N_times = set_vars(self.N_field, self.N_text, 'N = ')

        if self.dialog is not None:
            self.dialog.ShowModal()
        self.SIR = SIR_fit.SIR_fit(self.dt, self.T, self.S0, self.I0, self.R0, self.N_times)

    def plot_simulated_data(self,evt):
        if not self.data_generated:
            self.data = self.SIR.generate_data(self.beta0, self.gamma0)
            self.data_generated = True
        measurement_times = np.linspace(0, self.T, self.N_times)
        pp.plot(measurement_times, self.data,'.')
        pp.title('Simulated data points of I(t)')
        pp.xlabel('t')
        pp.ylabel('I(t)')
        pp.show()

    @timeit
    def fit_selection(self,evt):
        if self.dropdown.GetSelection() == 0:
            try:
                self.fit_SIR_model_gradient()
            except:
                self.dialog = wx.MessageDialog(None, message="Oops, something went wrong. Please try again.",
                                               style=wx.OK)
                self.dialog.ShowModal()
        elif self.dropdown.GetSelection() == 1:
            try:
                self.fit_SIR_model_CG()
            except:
                self.fit_selection(wx.EVT_BUTTON)
                print('Encountered error; had to restart')

    def fit_SIR_model_CG(self):
        if not self.data_generated:
            print('simulating data...')
            self.data = self.SIR.generate_data(self.beta0, self.gamma0)
            self.data_generated = True
            print('Done.')
        print('fitting')
        beta, gamma, residue = self.SIR.fit_model_CG(self.data)

        print('Done.')
        print('computing model...')
        S_fitted, I_fitted, R_fitted = self.SIR.compute_full_model(beta, gamma)
        print('Done.')
        self.plot_results(S_fitted, I_fitted, R_fitted, beta, gamma, residue, 'Conjugate Gradient')

    def fit_SIR_model_gradient(self):
        if not self.data_generated:
            print('simulating data...')
            self.data = self.SIR.generate_data(self.beta0, self.gamma0)
            self.data_generated = True
            print('Done.')
        print('fitting')
        beta, gamma, residue = self.SIR.fit_model_gradient(self.data)

        print('Done.')
        print('computing model...')
        S_fitted, I_fitted, R_fitted = self.SIR.compute_full_model(beta, gamma)
        print('Done.')
        self.plot_results(S_fitted, I_fitted, R_fitted, beta, gamma, residue, 'Gradient Descent')

    def plot_results(self, S_fitted, I_fitted, R_fitted, beta, gamma, residue, figure_title):

        fig = pp.figure(constrained_layout=True, figsize=(10,5))
        pp.gcf().canvas.set_window_title(figure_title)

        gs = GridSpec(2, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[1, 0:2])
        ax4 = fig.add_subplot(gs[1, 2:4])

        measurement_times = np.linspace(0, self.T, self.N_times)
        numpoints = int(np.ceil(self.T / self.dt))
        numeric_grid = np.linspace(0, self.T, numpoints)

        ax1.plot(measurement_times, self.data, '.')
        ax1.plot(numeric_grid, I_fitted, color='r')
        ax1.set_title('Data points and fit of I(t)')
        ax1.set_xlabel('t')

        ax2.plot(np.log(residue))
        ax2.set_title('Log of residual error')
        ax2.set_xlabel('gradient steps')

        ax3.plot(numeric_grid, S_fitted, label='S')
        ax3.plot(numeric_grid, I_fitted, label='I')
        ax3.plot(numeric_grid, R_fitted, label='R')
        ax3.set_title('Susceptibles, Infectious, Removed')
        ax3.legend(loc='center right')
        ax3.set_xlabel('t')

        ax4.plot()
        pp.axis('off')
        row_labels = ['Actual value for beta', 'Actual value for gamma',
                      'Estimated value for beta','Estimated value for gamma','Residual error',]
        table_vals = [[self.beta0],
                      [self.gamma0],
                      [round(beta,3)],
                      [round(gamma,3)],
                      [round(residue[-1],4)]]
        the_table = pp.table(cellText=table_vals,
                              colWidths=[0.3] * 2,
                              rowLabels=row_labels,
                              loc='center right')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(11)


        pp.show(block=False)

    def reset_app(self,evt):
        self.S0 = 1  # Initial value for "Susceptible"
        self.I0 = 0.01  # Initial value for "Infected"
        self.R0 = 0  # Initial value for "Removed"
        self.N_times = 150  # Times at which measurements are taken
        self.beta0 = 0.4
        self.gamma0 = 0.1
        self.T = 100
        self.dt = 0.01
        self.data_generated = False
        self.S_text.SetLabel('S0 = ' + str(self.S0))
        self.I_text.SetLabel('I0 = ' + str(self.I0))
        self.R_text.SetLabel('R0 = ' + str(self.R0))
        self.b_text.SetLabel('beta0 = ' + str(self.beta0))
        self.g_text.SetLabel('gamma0 = ' + str(self.gamma0))
        self.N_text.SetLabel('N = ' + str(self.N_times))
        self.SIR = SIR_fit.SIR_fit(self.dt, self.T, self.S0, self.I0, self.R0, self.N_times)


#=======================================================================================================


class MainApp(wx.App):
    def OnInit(self):
        frame = SIR_GUI()
        frame.Show()
        return True

app=MainApp()
app.MainLoop()