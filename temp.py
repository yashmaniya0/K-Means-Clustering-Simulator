import dash_core_components as dcc
import dash_html_components as html

# COLOR PALETTES
cluster_colors = {0: '#6395EC', 1: '#FFA27A', 2: '#DDA618', 3: '#DFA5DD',
                  4: '#61CCAB', 5: '#CD5C5C', 6: '#719D2F', 7: '#8E4D2C', 8: '#472F63'}
line_colors = {0: 'rgba(99,149,236, .7)', 1: 'rgba(255,162,122, .7)', 2: 'rgba(221,166,24, .7)', 3: 'rgba(223,165,221, .7)',
               4: 'rgba(97,204,171, .7)', 5: 'rgba(205,92,92, .7)', 6: 'rgba(113,157,47, .7)', 7: 'rgba(142,77,44, .7)',
               8: 'rgba(71,47,99, .7)'}

# APP LAYOUT

main_title = html.Div([
    html.H1('K-Means Clustering Simulator', style={'color': '#4B4D55', 'font-family': 'savoye LET',
                                                   'fontSize': 40, 'fontWeight': 100, 'textAlign': 'center', 'marginTop': '1%', 'marginBottom': '0%'})
], style={'height': '10%', 'width': '100%', 'border-radius': 10, 'borderStyle': 'solid', 'borderWidth': 2, 'borderColor': 'black', 'background-color': 'white'}
)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------


shape_filter = html.Div([
    html.Div('Shape :', style={'marginLeft': 2}),
    dcc.Dropdown(
        id='shape-dropdown',
        options=[
            {'label': 'Gaussian Mixture', 'value': 'gmm'},
            {'label': 'Circles', 'value': 'circle'},
            {'label': 'Moons', 'value': 'moon'},
            {'label': 'Anisotropicly distributed',
             'value': 'anisotropic'},
            {'label': 'No Structure', 'value': 'noStructure'},
        ],
        value='gmm'
    )
],  style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 15})

sample_size_filter = html.Div([
    html.Div('Sample Size :', style={'marginLeft': 2}),
    dcc.Slider(id='sample-slider', min=50, max=500,
               step=50, value=200, marks={50: '50', 500: '500'})
], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 25})

n_clusters_filter = html.Div([
    html.Div('Nb of Clusters :', style={'marginLeft': 2}),
    dcc.Slider(id='cluster-slider', min=2, max=9,
               marks={i: '{}'.format(i) for i in range(2, 10)}, value=3)
], style={'marginLeft': 10, 'marginRight': 10})

generate_data_btn = html.Div([
    html.Button('GENERATE DATA', id='regenData-button', style={
        'width': 130, 'font-size': 10, 'text-align': 'center', 'padding': 0}),
], style={'marginLeft': '22%', 'marginTop': 40})


init_method_filter = html.Div([
    html.Div('Initialization Method :',
             style={'marginLeft': 2}),
    dcc.Dropdown(
        id='init-dropdown',
        options=[
            {'label': 'Random', 'value': 'random'},
            {'label': 'K-means++', 'value': 'kmeans++'}
        ],
        value='random'
    )
],  style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 15})

n_centroids_filter = html.Div([
    html.Div('Nb of Centroids :', style={'marginLeft': 2}),
    dcc.Slider(id='centroid-slider', min=2, max=9,
               marks={i: '{}'.format(i) for i in range(2, 10)}, value=3)
], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 30})

max_iter_filter = html.Div([
    html.Div('Max Iterations :', style={'marginLeft': 2}),
    dcc.Slider(id='iter-slider', min=5, max=20,
               step=1, value=10, marks={5: '5', 20: '20'})
], style={'marginLeft': 10, 'marginRight': 10, 'marginBottom': 15})

generate_centroids_btn = html.Div([
    html.Button('GENERATE CENTROIDS', id='regenCentroids-button', style={
        'width': 130, 'font-size': 8, 'text-align': 'center', 'padding': 0}),
], style={'marginLeft': '22%', 'marginTop': 30})

main_plot = html.Div([
    html.H6('K-MEANS SCATTER PLOT', style={'text-align': 'center', 'marginTop': 30,
                                           'marginBottom': 5, 'fontSize': 15, 'fontWeight': 800, 'color': '#4B4D55'}),

    html.Button('Play', id='play-button', style={'marginLeft': 45, 'marginTop': 30, 'marginBottom': 10,
                                                 'marginRight': 5, 'width': '5%', 'font-size': 10, 'text-align': 'center', 'padding': 0}),
    html.Button('Pause', id='pause-button', style={
        'marginRight': 5, 'width': '8%', 'font-size': 10, 'text-align': 'center', 'padding': 0}),
    html.Button('<<', id='prevStep-button', style={
        'marginRight': 5, 'width': '5%', 'font-size': 10, 'text-align': 'center', 'padding': 0}),
    html.Button('>>', id='nextStep-button', style={
        'marginRight': 5, 'width': '5%', 'font-size': 10, 'text-align': 'center', 'padding': 0}),
    html.Button('Restart', id='restart-button',
                style={'width': '10%', 'font-size': 10, 'text-align': 'center', 'padding': 0}),

    html.Button(id='iter-text', disabled=True, style={'marginLeft': '14.5%', 'width': '22%', 'background-color': '#4B4D55',
                                                      'pointer-events': 'none', 'color': 'white', 'font-size': 10, 'text-align': 'center', 'padding': 0}),

    dcc.Graph(id='kmeans-graph', animate=True,
              config={'displayModeBar': False}, style={'marginLeft': 5, 'marginRight': 15}),
    dcc.Interval(id='interval-component',
                 interval=3600*1000, n_intervals=0),
    html.Div([], style={'height': '10%',
                        'top': '0%', 'position': 'relative'})

], className='nine columns', style={'background-color': 'white', 'height': '100%', 'width': '100%', 'border-width': 2, 'borderStyle': 'solid', 'border-radius': 10, 'borderColor': 'black', 'margin': 0})


real_sidebar = html.Div([
    # CREATE DATASET
    html.Div([
        html.H6('CREATE DATASET', style={'text-align': 'center', 'marginTop': 15,
                                         'marginBottom': 15, 'fontSize': 15, 'fontWeight': 800, 'color': '#4B4D55'}),
        shape_filter,
        sample_size_filter,
        n_clusters_filter,
        generate_data_btn,
    ]),

    html.Hr(style={'marginTop': 25, 'marginBottom': 15}),

    # K-MEANS INITIALIZATION
    html.Div([
        html.H6('INITIALIZE K-MEANS', style={'text-align': 'center',
                                             'marginBottom': 15, 'fontSize': 15, 'fontWeight': 800, 'color': '#4B4D55'}),
        init_method_filter,
        n_centroids_filter,
        max_iter_filter,
        generate_centroids_btn,
    ]),
], className='three columns', style={'background-color': 'white', 'height': '100%', 'width': '100%', 'borderStyle': 'solid', 'border-radius': 10, 'borderWidth': 2, 'borderColor': 'black'})


inertia_plot = html.Div([
    html.H6('K-MEANS COST FUNCTION', style={'text-align': 'center', 'marginTop': 15,
                                            'marginBottom': 5, 'fontSize': 15, 'fontWeight': 800, 'color': '#4B4D55'}),
    dcc.Graph(id='inertia-graph', config={'displayModeBar': False}, style={
        'marginLeft': 5, 'marginRight': 5}),
], style={'background-color': 'white', 'height': '49%', 'border-radius': 10, 'borderStyle': 'solid', 'borderWidth': 2, 'borderColor': 'black'})


silhouette_plot = html.Div([
    html.H6('SILHOUETTE ANALYSIS', style={'text-align': 'center', 'marginTop': 15,
                                          'marginBottom': 5, 'fontSize': 15, 'fontWeight': 800, 'color': '#4B4D55'}),
    dcc.Graph(id='silhouette-graph', config={'displayModeBar': False}, style={
        'marginLeft': 5, 'marginRight': 5})
], style={'background-color': 'white', 'height': '49%', 'position': 'relative', 'top': '1.6%', 'border-radius': 10, 'borderStyle': 'solid', 'borderWidth': 2, 'borderColor': 'black'})


hidden_divs = [
    # HIDDEN DIVISIONS FOR STORING INTERMEDIATE DATA
    html.Div(id='dataset-value', style={'display': 'none'}),
    html.Div(id='kmeansCentroids-value', style={'display': 'none'}),
    html.Div(id='kmeans_frames-counter', style={'display': 'none'}),
]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

final_layout = html.Div(children=[

    # COLUMN 1 (PARAMETERS BLOCK)
    html.Div([
        real_sidebar
    ], className='eight columns', style={'height': '96%', 'position': 'relative', 'top': '1%', 'width': '20%'}),

    # COLUMN 2 (TITLE + K-MEANS GRAPH)
    html.Div([
        # ROW 1 (TITLE)
        main_title,
        # ROW 2 (GRAPH)
        main_plot
    ], className='eight columns', style={'height': '96%', 'position': 'relative', 'top': '1%', 'width': '32.5%'}),

    # COLUMN 3 (COST FUNCTION GRAPH + SILHOUETTE GRAPH)
    html.Div([
        inertia_plot,
        silhouette_plot,
    ], className='four columns', style={'height': '96%', 'marginLeft': '1.5%', 'width': '32.5%'}),

    html.Div(hidden_divs, className='twelve columns')
], style={'height': 820})

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------


layout3 = html.Div(children=[

    # HEADER
    html.Div([
        html.H1('K-means Clustering Visualization',
                style={'color': '#333', 'textAlign': 'center'})
    ], style={'backgroundColor': '#F4F4F4', 'padding': '20px 0', 'marginBottom': '20px'}),

    # MAIN CONTENT
    html.Div([
        # LEFT SIDEBAR
        html.Div([
            html.H3('Parameters', style={'color': '#333'}),
            # Dropdowns and Sliders for parameters
            # ...

            html.Button('Generate Data', id='regenData-button',
                        style={'marginTop': '20px', 'width': '100%'}),
            html.Button('Generate Centroids', id='regenCentroids-button',
                        style={'marginTop': '20px', 'width': '100%'}),
        ], className='four columns', style={'backgroundColor': '#F9F9F9', 'padding': '20px', 'marginRight': '20px'}),

        # MAIN GRAPH AREA
        html.Div([
            # K-means scatter plot
            dcc.Graph(id='kmeans-graph', animate=True,
                      config={'displayModeBar': False}),
            html.Div(id='iter-text',
                     style={'textAlign': 'center', 'marginTop': '10px'})
        ], className='eight columns', style={'backgroundColor': '#FFF', 'padding': '20px', 'borderRadius': '5px'})
    ], className='row', style={'marginBottom': '20px'}),

    # ADDITIONAL GRAPHS
    html.Div([
        # Inertia graph
        html.Div([
            html.H3('K-means Cost Function', style={'color': '#333'}),
            dcc.Graph(id='inertia-graph', config={'displayModeBar': False})
        ], className='six columns', style={'backgroundColor': '#F9F9F9', 'padding': '20px', 'marginRight': '20px'}),

        # Silhouette graph
        html.Div([
            html.H3('Silhouette Analysis', style={'color': '#333'}),
            dcc.Graph(id='silhouette-graph', config={'displayModeBar': False})
        ], className='six columns', style={'backgroundColor': '#F9F9F9', 'padding': '20px'})
    ], className='row')
])
