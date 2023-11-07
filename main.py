# Main module for .dat file data import



# Import the numeric modules.
import numpy as np
import scipy.io.matlab as matlab
import base64
from io import BytesIO

from random import random

# Signal information
from experiment.data import SingleExperimentData

# Plotting.
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as splt
import dash



# Global collection of all loaded signals in a session.
# This is a requirement because Dash does not allow non serializable objects in Store.
signals = dict()


def webview():
    """Creates the WebView application to parse the data."""

    # Create the application.
    app = dash.Dash(__name__)

    # Create the layout.
    app.layout = dash.html.Div(
        style={
            'padding': '0',
        },
        children=[
            # Left side topbar.
            dash.html.Div(
                style={
                    'display': 'inline-block',
                    'width': '50%',
                    'margin': '0',
                    'padding': '0',
                },
                children=[
                    dash.html.H2("Interactive analysis plot application"),

                    # Menu setting to select a one or two signal plot.
                    dash.html.Div(
                        children=[
                            'Number of signals',
                            dash.dcc.RadioItems(
                                ['One', 'Two'],
                                'One',
                                id='select-number',
                                inline=True,
                            ),
                        ],
                        style={
                            'width': '30%',
                            'display': 'inline-block',
                        },
                    ),

                    # Menu setting to select type of graph.
                    dash.html.Div(
                        children=[
                            'Graph type',
                            dash.dcc.Dropdown(
                                ['Signal', 'FFT', 'Hilbert'],
                                'Signal',
                                id='select-graph',
                            ),
                        ],
                        style={
                            'width': '30%',
                            'display': 'inline-block',
                        },
                    ),

                    # Menu setting to select compare or diff when dual.
                    dash.html.Div(
                        children=[
                            'Comparison type',
                            dash.dcc.RadioItems(
                                ['Split', 'Overlap'],
                                'Split',
                                id='select-compare',
                                inline=True,
                            ),
                        ],
                        style={
                            'width': '30%',
                            'display': 'inline-block',
                        },
                    ),

                    # Upload new signals.
                    dash.dcc.Upload(
                        id='upload-data',
                        multiple=True,
                        children=dash.html.Div([
                            'Drag and Drop or ',
                            dash.html.A('Select Files'),
                            ' to add signals'
                        ]),
                        style={
                            'width': '90%',
                            'height': '50px',
                            'lineHeight': '50px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '5px',
                        },
                    ),
                ],
            ),

            # Menu setting to select the signals.
            dash.html.Div(
                style={
                    'display': 'inline-block',
                    'margin': '0',
                    'width': '48%',
                },
                children=[
                    dash.html.Div(
                        children=[
                            'Signal A',
                            dash.html.Div(
                                children=[
                                    'Experiment',
                                    dash.dcc.Dropdown(id='expa-select'),
                                ],
                                style={
                                    'width': '45%',
                                },
                            ),
                            dash.html.Div(
                                children=[
                                    'Signal',
                                    dash.dcc.Dropdown(id='siga-select'),
                                ],
                                style={
                                    'width': '45%',
                                },
                            ),
                        ],
                        style={
                            'width': '48%',
                            'display': 'inline-block',
                        },
                    ),

                    dash.html.Div(
                        children=[
                            'Signal B',
                            dash.html.Div(
                                children=[
                                    'Experiment',
                                    dash.dcc.Dropdown(id='expb-select'),
                                ],
                                style={
                                    'width': '45%',
                                },
                            ),
                            dash.html.Div(
                                children=[
                                    'Signal',
                                    dash.dcc.Dropdown(id='sigb-select'),
                                ],
                                style={
                                    'width': '45%',
                                },
                            ),
                        ],
                        style={
                            'width': '48%',
                            'display': 'inline-block',
                        },
                    ),
                ],
            ),

            # Configured graph.
            dash.dcc.Graph(
                id='main-graph',
                style={
                    'height': '90vh',
                    'margin': '0',
                    'padding': '0',
                },
            ),

            # Storage for the processed data.
            dash.dcc.Store(id='signal-storage'),

            # Storage for the names of the experiments.
            dash.dcc.Store(id='expname-storage'),

            # Storage for the current graph configuration.
            dash.dcc.Store(id='graph-config'),
        ],
    )

    app.run(threaded=True)



@dash.callback(
    dash.Output('signal-storage', 'value'),
    dash.Input('upload-data', 'contents'),
    dash.State('upload-data', 'filename'),
)
def upload(contents: list, names: list[str]):
    """Upload file handler"""

    # Declare the use of the global variable signals.
    global signals

    # Initialize signals if it's None.
    if signals is None:
        signals = dict()

    # Check that something was uploaded.
    if (contents is None) or (names is None):
        raise dash.exceptions.PreventUpdate

    # Process all uploaded files.
    for name, content in zip(names, contents):
        # Check if the name of the signal is in the dictionary.
        if name in signals.keys():
            raise dash.exceptions.PreventUpdate

        # Insert the filename in the dictionary.
        signals[name] = dict()

        # Get the contents.
        _, encoded = content.split(',')

        # Decode from base 64.
        decoded = BytesIO( base64.b64decode( encoded ) )

        ## Parse the Matlab file.
        ## This section is copied from the scipy source code.
        ## ####################################################################

        # Get the Matlab major version.
        mjv, mnv = matlab._miobase._get_matfile_version( decoded )

        # Check if the file has a correct version.
        if mjv == 0:
            reader = matlab._mio.MatFile4Reader( decoded )
        elif mjv == 1:
            reader = matlab._mio.MatFile5Reader( decoded )
        elif mjv == 2:
            raise NotImplementedError('Unsupported HDF reader')
        else:
            raise TypeError(f'Did not recognize Matlab version {mjv}')

        # Get the internal variables.
        matfile = reader.get_variables(None)

        ## ####################################################################

        # Parse the contents of the Matlab data and anlyze them.
        experiment = SingleExperimentData.parse( matfile )
        experiment.analyze()

        # Add the experiment header to the storage.
        signals[name]['header'] = experiment

        # Add all signals of this experiment to the storage.
        for signal in experiment.signals:
            # Create the name of the signal.
            signame = f"A{signal.actuator.id} S{signal.sensor.id} {signal.actuator.freq / 1000.0} kHz {signal.actuator.volts} V"

            # Add this signal to the dictionary.
            signals[name][signame] = signal

    return random()


@dash.callback(
    dash.Output('expa-select', 'options'),
    dash.Input('signal-storage', 'value'),
)
def expa(ignore):
    """Updates the options list of the Signal A experiment selection."""

    # Declare the use of the global variable signals.
    global signals

    # Do not update for None.
    if signals is None:
        raise dash.exceptions.PreventUpdate

    return sorted( signals.keys() )



@dash.callback(
    dash.Output('siga-select', 'options'),
    dash.Input('expa-select', 'value'),
)
def siga(exp: str | None):
    """Updates the options list of the Signal A signal selection."""

    # Declare the use of the global variable signals.
    global signals

    # Do not update for None.
    if (exp is None) or (signals is None):
        raise dash.exceptions.PreventUpdate

    # Return the list of signal names in the given experiment.
    return sorted( signals[exp].keys() )



@dash.callback(
    dash.Output('expb-select', 'options'),
    dash.Input('signal-storage', 'value'),
)
def expb(ignore):
    """Updates the options list of the Signal B experiment selection."""

    # Declare the use of the global variable signals.
    global signals

    # Do not update for None.
    if signals is None:
        raise dash.exceptions.PreventUpdate

    return sorted( signals.keys() )



@dash.callback(
    dash.Output('sigb-select', 'options'),
    dash.Input('expb-select', 'value'),
)
def sigb(exp: str | None):
    """Updates the options list of the Signal B signal selection."""

    # Declare the use of the global variable signals.
    global signals

    # Do not update for None.
    if (exp is None) or (signals is None):
        raise dash.exceptions.PreventUpdate

    # Return the list of signal names in the given experiment.
    return sorted( signals[exp].keys() )



def buildone(config: dict):
    """Creates the graph for one signal."""

    # Declare the use of the global variable signals.
    global signals

    # Check that both epxeriment and signal are selected.
    if (config['expa'] is None) or (config['siga'] is None):
        raise dash.exceptions.PreventUpdate

    # Get the signal requested.
    signal = signals[config['expa']][config['siga']]

    # Create the signal graph.
    if config['graph'] == 'Signal':
        # Get the X axis for all the arrays.
        x = pd.Series( np.array( range( 0, len( signal.sen.flatten() ) ) ) * signal.timestep )

        # Create the figure.
        fig = go.Figure()

        # Add all the signals.
        fig.add_scatter( x=x, y=pd.Series( signal.sen.flatten() ), name='Sensor signal [filter]' )
        fig.add_scatter( x=x, y=pd.Series( signal.act.flatten() ), name='Source signal [smoothed]' )
        fig.add_scatter( x=x, y=pd.Series( signal.raw.flatten() ), name='Sensor signal [raw]' )
        fig.add_vline( signal.endSource   * signal.timestep, line={'color': "#000000"} )
        fig.add_vline( signal.startSignal * signal.timestep, line={'color': "#FF00FF"} )
        fig.add_vline( signal.sigmax      * signal.timestep, line={'color': "#00FF00"} )

        # Set the title
        fig.update_layout(
            title=dict(text=f"{config['siga']} signal", font=dict(size=25), yref='paper'),
            xaxis_title="T [us]",
            yaxis_title="Voltage [V]"
        )

    # Create the Hilbert envelope.
    elif config['graph'] == 'Hilbert':
        # Create the figure.
        fig = go.Figure()

        # Get the X axis for all the arrays.
        x = pd.Series( np.array( range( 0, len( signal.sen.flatten() ) ) ) * signal.timestep )

        # Add all the signals.
        fig.add_scatter( x=x, y=pd.Series( signal.hil.flatten() ), name='Sensor envelope [Hilbert]' )
        fig.add_vline( signal.endSource   * signal.timestep, line={'color': "#000000"} )
        fig.add_vline( signal.startSignal * signal.timestep, line={'color': "#FF00FF"} )
        fig.add_vline( signal.sigmax      * signal.timestep, line={'color': "#00FF00"} )

        # Set the title
        fig.update_layout(
            title=dict(text=f"{config['siga']} envelope", font=dict(size=25), yref='paper'),
            xaxis_title="T [us]",
            yaxis_title="Voltage [V]"
        )

    # Create the heatmap graph.
    else:
        # Create the heatmap.
        heatmap = go.Heatmap( x=signal.fftT, y=np.divide( signal.fftF, 1000 ), z=signal.fftZ )

        # Create the figure.
        fig = go.Figure( heatmap )

        # Set the title
        fig.update_layout(
            title=dict(text=f"{config['siga']} heatmap", font=dict(size=25), yref='paper'),
            xaxis_title="T [us]",
            yaxis_title="Frequency [kHz]"
        )

    return fig


def buildtwo(config: dict):
    """Creates the graph for two signals."""

    # Declare the use of the global variable signals.
    global signals

    # Check that both epxeriment and signal are selected.
    if (config['expa'] is None) or (config['siga'] is None) or (config['expb'] is None) or (config['sigb'] is None):
        raise dash.exceptions.PreventUpdate

    # Get the signal requested.
    signala = signals[config['expa']][config['siga']]
    signalb = signals[config['expb']][config['sigb']]

    # Create the overlap graph.
    if config['compare'] == 'Overlap':
        # Create the signal graph.
        if config['graph'] == 'Signal':

            # Get the X axis for all the arrays.
            xa = pd.Series( np.array( range( 0, len( signala.sen.flatten() ) ) ) * signala.timestep )
            xb = pd.Series( np.array( range( 0, len( signalb.sen.flatten() ) ) ) * signalb.timestep )

            # Create the figure.
            fig = go.Figure()

            # Add all the signals.
            fig.add_scatter( x=xa, y=pd.Series( signala.sen.flatten() ), name='Sensor A signal [filter]' )
            fig.add_scatter( x=xa, y=pd.Series( signala.act.flatten() ), name='Source A signal [smoothed]' )
            fig.add_scatter( x=xa, y=pd.Series( signala.raw.flatten() ), name='Sensor A signal [raw]' )
            fig.add_scatter( x=xb, y=pd.Series( signalb.sen.flatten() ), name='Sensor B signal [filter]' )
            fig.add_scatter( x=xb, y=pd.Series( signalb.act.flatten() ), name='Source B signal [smoothed]' )
            fig.add_scatter( x=xb, y=pd.Series( signalb.raw.flatten() ), name='Sensor B signal [raw]' )

            # Add the timestamps.
            fig.add_vline( signala.endSource   * signala.timestep, line={'color': "#000000"} )
            fig.add_vline( signala.startSignal * signala.timestep, line={'color': "#FF00FF"} )
            fig.add_vline( signalb.endSource   * signalb.timestep, line={'color': "#000000"} )
            fig.add_vline( signalb.startSignal * signalb.timestep, line={'color': "#00FF00"} )

            # Set the title
            fig.update_layout(
                title=dict(text=f"Signals of {config['siga']} and {config['sigb']}", font=dict(size=25), yref='paper'),
                xaxis_title="T [us]",
                yaxis_title="Voltage [mV]"
            )

        # Create the Hilbert envelope.
        elif config['graph'] == 'Hilbert':
            # Create the figure.
            fig = go.Figure()

            # Get the X axis for all the arrays.
            x = pd.Series( np.array( range( 0, len( signala.hil.flatten() ) ) ) * signala.timestep )

            # Add all the signals.
            fig.add_scatter( x=x, y=pd.Series( signala.hil.flatten()                         ), name='Sensor A delta [Hilbert]' )
            fig.add_scatter( x=x, y=pd.Series(                         signalb.hil.flatten() ), name='Sensor B delta [Hilbert]' )
            fig.add_scatter( x=x, y=pd.Series( signala.hil.flatten() - signalb.hil.flatten() ), name='Envelope delta [Hilbert]' )

            # Add the timestamps.
            fig.add_vline( signala.endSource   * signala.timestep, line={'color': "#000000"}, row=1, col=1 )
            fig.add_vline( signala.startSignal * signala.timestep, line={'color': "#FF00FF"}, row=1, col=1 )
            fig.add_vline( signalb.endSource   * signalb.timestep, line={'color': "#000000"}, row=2, col=1 )
            fig.add_vline( signalb.startSignal * signalb.timestep, line={'color': "#00FF00"}, row=2, col=1 )

            # Set the title
            fig.update_layout(
                title=dict(text=f"{config['siga']} envelope", font=dict(size=25), yref='paper'),
                xaxis_title="T [us]",
                yaxis_title="Voltage [mV]"
            )

        # Create the heatmap graph.
        else:
            # Create the heatmap.
            heatmap = go.Heatmap( x=signala.fftT, y=np.divide( signala.fftF, 1000 ), z=signala.fftZ - signalb.fftZ )

            # Create the figure.
            fig = go.Figure( heatmap )

            # Set the title
            fig.update_layout(
                title=dict(text=f"Heatmap Diff {config['siga']} - {config['sigb']}", font=dict(size=25), yref='paper'),
                xaxis_title="T [us]",
                yaxis_title="Frequency [kHz]"
            )

    # Create the split graphs.
    else:
        # Create the signal graph.
        if config['graph'] == 'Signal':
            # Get the signal requested.
            signala = signals[config['expa']][config['siga']]
            signalb = signals[config['expb']][config['sigb']]

            # Get the X axis for all the arrays.
            xa = pd.Series( np.array( range( 0, len( signala.sen.flatten() ) ) ) * signala.timestep )
            xb = pd.Series( np.array( range( 0, len( signalb.sen.flatten() ) ) ) * signalb.timestep )

            # Create the figure.
            fig = splt.make_subplots(rows=2, cols=1, shared_xaxes=True)

            # Calculate the min and max of the graphs.
            ymin = np.min([
                np.min(signalb.sen.flatten()), np.min( signala.sen.flatten() ),
                np.min(signalb.raw.flatten()), np.min( signala.raw.flatten() ),
            ])

            ymax = np.max([
                np.max(signalb.sen.flatten()), np.max( signala.sen.flatten() ),
                np.max(signalb.raw.flatten()), np.max( signala.raw.flatten() ),
            ])

            # Fix the Y axis ranges.
            fig.update_yaxes( range=[ymin, ymax], row=1, col=1 )
            fig.update_yaxes( range=[ymin, ymax], row=2, col=1 )

            # Add all the signals.
            fig.add_scatter( x=xa, y=pd.Series( signala.sen.flatten() ), name='Sensor A signal [filter]'  , row=1, col=1 )
            fig.add_scatter( x=xa, y=pd.Series( signala.act.flatten() ), name='Source A signal [smoothed]', row=1, col=1 )
            fig.add_scatter( x=xa, y=pd.Series( signala.raw.flatten() ), name='Sensor A signal [raw]'     , row=1, col=1 )
            fig.add_scatter( x=xb, y=pd.Series( signalb.sen.flatten() ), name='Sensor B signal [filter]'  , row=2, col=1 )
            fig.add_scatter( x=xb, y=pd.Series( signalb.act.flatten() ), name='Source B signal [smoothed]', row=2, col=1 )
            fig.add_scatter( x=xb, y=pd.Series( signalb.raw.flatten() ), name='Sensor B signal [raw]'     , row=2, col=1 )

            # Add the timestamps.
            fig.add_vline( signala.endSource   * signala.timestep, line={'color': "#000000"}, row=1, col=1 )
            fig.add_vline( signala.startSignal * signala.timestep, line={'color': "#FF00FF"}, row=1, col=1 )
            fig.add_vline( signalb.endSource   * signalb.timestep, line={'color': "#000000"}, row=2, col=1 )
            fig.add_vline( signalb.startSignal * signalb.timestep, line={'color': "#00FF00"}, row=2, col=1 )

            fig.update_layout(
                title=dict(text=f"Signals {config['siga']} and {config['sigb']}", font=dict(size=25), yref='paper'),
                xaxis_title="T [us]",
                yaxis_title="Voltage [mV]"
            )

        # Create the Hilbert envelope.
        elif config['graph'] == 'Hilbert':
            # Create the figure.
            fig = splt.make_subplots(rows=2, cols=1, shared_xaxes=True)

            # Get the X axis for all the arrays.
            xa = pd.Series( np.array( range( 0, len( signala.hil.flatten() ) ) ) * signala.timestep )
            xb = pd.Series( np.array( range( 0, len( signalb.hil.flatten() ) ) ) * signalb.timestep )

            # Add all the signals.
            # Add all the signals.
            fig.add_scatter( x=xa, y=pd.Series( signala.hil.flatten() ), name='Sensor A envelope [Hilbert]', row=1, col=1 )
            fig.add_scatter( x=xb, y=pd.Series( signalb.hil.flatten() ), name='Sensor B envelope [Hilbert]', row=2, col=1 )

            # Add the timestamps.
            fig.add_vline( signala.endSource   * signala.timestep, line={'color': "#000000"}, row=1, col=1 )
            fig.add_vline( signala.startSignal * signala.timestep, line={'color': "#FF00FF"}, row=1, col=1 )
            fig.add_vline( signalb.endSource   * signalb.timestep, line={'color': "#000000"}, row=2, col=1 )
            fig.add_vline( signalb.startSignal * signalb.timestep, line={'color': "#00FF00"}, row=2, col=1 )

            # Set the title
            fig.update_layout(
                title=dict(text=f"{config['siga']} and {config['sigb']} envelopes [Hilbert]", font=dict(size=25), yref='paper'),
                xaxis_title="T [us]",
                yaxis_title="Voltage [mV]"
            )

        # Create the heatmap graph.
        else:
            # Get the maximum Z value.
            zmax = max( np.max( signala.fftZ.flatten() ), np.max( signalb.fftZ.flatten() ) )

            # Create the heatmap.
            heatmapa = go.Heatmap( x=signala.fftT, y=np.divide( signala.fftF, 1000 ), z=signala.fftZ, autocolorscale=False, zmin=0, zmax=zmax )
            heatmapb = go.Heatmap( x=signalb.fftT, y=np.divide( signalb.fftF, 1000 ), z=signalb.fftZ, autocolorscale=False, zmin=0, zmax=zmax )

            # Create the figure.
            fig = splt.make_subplots(rows=2, cols=1, shared_xaxes=True)

            # Populate the figure.
            fig.add_trace( heatmapa, row=1, col=1 )
            fig.add_trace( heatmapb, row=2, col=1 )

            # Set title and axis.
            fig.update_layout(
                title=dict(text=f"Heatmaps of {config['siga']} and {config['sigb']}", font=dict(size=25), yref='paper'),
                xaxis_title="T [us]",
                yaxis_title="Frequency [kHz]"
            )

    return fig


@dash.callback(
    dash.Output('main-graph', 'figure'),
    dash.Input('graph-config', 'value'),
)
def buildgraph(config: dict | None):
    """Callback to create an updated graph."""

    # Declare the use of the global variable signals.
    global signals

    # Check for an unconfigured graph.
    if (config is None) or (signals is None):
        raise dash.exceptions.PreventUpdate

    # Create the graph view for one signal.
    if config['number'] == 'One':
        return buildone(config)
    # Create the graph view for two signals.
    else:
        return buildtwo(config)



@dash.callback(
    dash.Output('graph-config' , 'value'),
    dash.Input('select-number' , 'value'),
    dash.Input('select-graph'  , 'value'),
    dash.Input('select-compare', 'value'),
    dash.Input('expa-select'   , 'value'),
    dash.Input('siga-select'   , 'value'),
    dash.Input('expb-select'   , 'value'),
    dash.Input('sigb-select'   , 'value'),
    dash.State('graph-config'  , 'value'),
)
def cfgupdate(number: str, graph: str, compare: str, expa: str, siga: str, expb: str, sigb: str, current: dict | None):
    # Ensure the current configuration is initialized.
    if current is None:
        current = dict([
            ('number' , 'One'   ),
            ('graph'  , 'Signal'),
            ('compare', 'Split' ),
            ('expa'   , None    ),
            ('siga'   , None    ),
            ('expb'   , None    ),
            ('sigb'   , None    ),
        ])

    # Checks which graph configuration fields are unchanged.
    changes = [
        number   != current['number'] ,
        graph    != current['graph']  ,
        compare  != current['compare'],
        expa     != current['expa']   ,
        siga     != current['siga']   ,
        expb     != current['expb']   ,
        sigb     != current['sigb']   ,
    ]

    # Check if there were any changes and avoid unnecessary updates.
    # Comparing to False because I dont trust what the fuck 'not' is doing.
    if any( changes ) == False:
        raise dash.exceptions.PreventUpdate

    # If the change is in signal B and number is one, do not emit changes.
    if (current['number'] == 'One') and (any( changes[0:5] ) == False):
        raise dash.exceptions.PreventUpdate

    # Update the dict.
    return dict([
        ('number' , number ),
        ('graph'  , graph  ),
        ('compare', compare),
        ('expa'   , expa   ),
        ('siga'   , siga   ),
        ('expb'   , expb   ),
        ('sigb'   , sigb   ),
    ])


if __name__ == "__main__":
    webview()
