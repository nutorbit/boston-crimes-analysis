import folium
import math
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from plotly.offline import iplot
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from utils.constants import BOSTON_LAT, BOSTON_LONG, COLORS # constants


def countNull(df):
    '''
    Count Null or NaN values in dataframe.
    
    Args:
        df (dataframe): dataframe that you want to count
        
    Returns:
        dataframe
    '''
    
    return pd.DataFrame(df.isnull().sum()*100/len(df), columns=['number_of_null(%)']).sort_values('number_of_null(%)', ascending=False)


def getLocations(df):
    '''
    Preprocess Lat & Long.
    
    Args:
        df (dataframe)
        
    Returns:
        dataframe 2 columns (Lat, Long)
    '''
    
    return df[['Lat', 'Long']].dropna().drop_duplicates().reset_index()[['Lat', 'Long']]


def plotMap(df, sample=500):
    
    m = folium.Map(location=[BOSTON_LAT, BOSTON_LONG], zoom_start=11, tiles='Stamen Toner',)

    def toCircle(x):
        return folium.Circle(
            radius=50,
            location=[x['Lat'], x['Long']],
            popup=f"{x['Lat']}, {x['Long']}",
            color='blue'
        ).add_to(m)

    _ = df.head(sample).apply(toCircle, axis=1)

    return m


def plotCluster(df, density=False):
    
    if not density:
    
        df_filter = df[(df['cluster'] < len(COLORS)) & (df['cluster'] >= 0)]

        m = folium.Map(location=[BOSTON_LAT, BOSTON_LONG], zoom_start=11, tiles='Stamen Toner',)

        def toCircle(x):
            return folium.Circle(
                radius=50,
                location=[x['Lat'], x['Long']],
                popup=f"{x['Lat']}, {x['Long']}",
                color=COLORS[int(x['cluster'])]
            ).add_to(m)

        _ = df_filter.apply(toCircle, axis=1)

        return m
    
    else:
        
        def getDistance(lat1, long1, lat2, long2):
            '''
            Get the distance between 2 point in lat, long coordinates.
            '''

            lat1   = math.radians(lat1)
            lat2   = math.radians(lat2)

            d_lat  = math.radians(lat1 - lat2)
            d_long = math.radians(long1 - long2)

            a = math.sin(d_lat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_long/2)**2

            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

            d = 6371 * c

            return d * 1000 # kilometer to meter
    

        # grouping each cluster and find center lat & long & size
        def info_group(x):
            d = {
                'center_lat': np.mean(x['Lat']),
                'center_long': np.mean(x['Long']),
                'diameter': getDistance(min(x['Lat']), min(x['Long']), max(x['Lat']), max(x['Long'])),
                'size': len(x['Lat'])
            }
            return pd.Series(d)

        cluster_group = df.groupby('cluster').apply(info_group).reset_index()
        cluster_group = cluster_group.iloc[1:]
        
        cluster_group['size'] = MinMaxScaler().fit_transform(cluster_group[['size']]).ravel() # scaling

        m = folium.Map(location=[BOSTON_LAT, BOSTON_LONG], zoom_start=11, tiles='Stamen Toner',)

        # add drug area
        def toCircle(x):
            return folium.Circle(
                radius=x['diameter']/2,
                location=[x['center_lat'], x['center_long']],
                popup=f"size:{x['size']}<br> diameter:{x['diameter']}",
                color="blue",
                fill=True
            ).add_to(m)

        _ = cluster_group.apply(toCircle, axis=1)


        return m
    

def getCluster(df, eps=0.002, min_samples=10):
    '''
    Cluster with DBSCAN.
    
    Args:
        df (dataframe)
        eps (float): radius for DBSCAN
        min_samples (int): minimum per cluster
        
    Returns:
        dataframe
    '''
    
    # define DBSCAN and predict cluster
    m = DBSCAN(eps=eps, min_samples=min_samples)
    pred = m.fit_predict(df)
    
    # concat cluster to data
    df_concat = pd.concat(
        [df, pd.Series(pred)],
        axis=1
    )

    # re-columns
    df_concat.columns = list(df_concat.columns)[:-1] + ['cluster']
    
    # filter
    df_concat = df_concat[df_concat['cluster'] >= 0].reset_index().drop('index', axis=1)
    
    return df_concat
    
    

def plotDistance(df, size=10):
    '''
    Distance plot with KNN.
    
    Args:
        df (dataframe): dataframe that you want to plot
        size (int): size of neighbor of KNN
        
    Returns:
        plot
    '''
    
    m = NearestNeighbors(n_neighbors=size).fit(df)
    distances, _ = m.kneighbors(df)
    distances = sorted(distances[:,size-1])

    # plot
    data = [
        go.Scatter(
            x=list(range(1,len(df))),
            y=distances[:-1],
        )
    ]

    layout = go.Layout(
        title='Distance plot',
        yaxis=dict(
            title='distance'
        ),
        xaxis=dict(
            title='data point'
        )
    )

    fig = go.Figure(data=data, layout=layout)

    return iplot(fig)