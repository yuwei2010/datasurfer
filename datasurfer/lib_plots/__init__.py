import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import abc
from functools import wraps
from datasurfer.lib_plots.plot_collection import plot_histogram, plot_dendrogram, plot_parallel_coordinate
from datasurfer.lib_plots.plot_utils import parallel_coordis, get_histo_bins
from datasurfer.datautils import parse_data

figparams = {'figsize': (8, 6), 
             'dpi': 105,}

def set_ax(ax):
    
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(which='major', ls='--')
    ax.grid(which='minor', ls=':')
    
    return ax
    
axisfunc = set_ax

def define_ax(func):
    """
    A decorator function that defines an axis for plotting.

    Parameters:
    - func: The function to be decorated.

    Returns:
    - The decorated function.

    Usage:
    - Use this decorator to define an axis for plotting in functions that require an axis.
    - If an axis is not provided as a keyword argument, a new axis will be created using plt.subplots().

    Example:
    @define_ax
    def plot_data(self, *keys, ax=None, **kwargs):
        # Function implementation
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs): 
          
        ax = kwargs.pop('ax', None)
        setax = kwargs.pop('setax', False)
        
        if not ax:
            _, ax = plt.subplots(**figparams)        
        
        ax = func(self, *args, ax=ax, **kwargs)
        
        if setax:
            axisfunc(ax)
        
        return ax
    return wrapper
    

    

class Plots(object):
    """
    A class for generating statistical plots.
    
    Parameters:
    - dp: A pandas DataFrame containing the data.
    """
    
    def __init__(self, db=None):
        """
        Initialize the Stat_Plots object.
        
        Parameters:
        - dp: A pandas DataFrame containing the data.
        """
        self.db = db
        
    def __getattr__(self, name: str):
        """
        Retrieve the value of an attribute dynamically.

        This method is called when an attribute is accessed that doesn't exist
        in the current object. It attempts to retrieve the attribute from the
        underlying database object and returns it.

        Parameters:
        - name (str): The name of the attribute to retrieve.

        Returns:
        - Any: The value of the attribute.

        Raises:
        - AttributeError: If the attribute doesn't exist in the current object
                            or the underlying database object.
        """
        try:
            return self.__getattribute__(name)
        
        except AttributeError:
            
            raise
                       
        
    def __call__(self, **kwargs):
        """
        Call the `line` method with the given key and keyword arguments.

        Parameters:
        - key: The key to identify the line.
        - **kwargs: Additional keyword arguments to pass to the `line` method.

        Returns:
        - The result of the `line` method.

        """

        return self.set_params(**kwargs)

    
    @property
    def pandas(self):
        
        """
        Further information at https://pandas.pydata.org/docs/reference/plotting.html
        
        
        """
        
        __all__ = [
            'area',
            'bar',
            'barh',
            'box',
            'density',
            'hexbin',
            'hist',
            'kde',
            'line',
            'pie',
            'scatter'
        ]
               
        class wrapper(object):           
            def __init__(self, db):
                self.db = db
                       
            def __getattr__(self, name):
                
                assert name in __all__, f'"{name}" is not a valid pandas plotting method. Please choose from {__all__}'
                
                def foo(*args, **kwargs):
                    @define_ax
                    @parse_data(add_labels=True)
                    def boo(self, *args, **kwargs):                       
                        labels = kwargs.pop('labels', None)
                        ax = kwargs.pop('ax', None)
                        if labels:
                            df = pd.DataFrame(dict(zip(labels, args)))
                            out = getattr(df.plot, name)(ax=ax, **kwargs)
                        else:
                            raise ValueError('Keys or dataframe must be provided')                        
                        return out
                    return boo(self, *args, **kwargs)  
                
                foo.__name__ = name     
              
                return foo  
                      
        return wrapper(self.db)
    
    @property
    def seaborn(self):
        """
        A method that provides a wrapper for seaborn plotting functions.

        Returns:
            wrapper: An instance of the wrapper class that allows access to seaborn plotting functions.
        """
        import seaborn as sns
        __all__ = [
            'clustermap',
            'jointplot',
            'pairplot',
            'swarmplot',
            'violinplot',
            'heatmap',
            'scatterplot',
            'lineplot',
            'boxplot',
            'kdeplot',
            'lmplot',
            'relplot',
            'catplot',
            'barplot',
            'countplot',
            'pointplot',
            'stripplot',
            'pairplot',
            'FacetGrid',
            'PairGrid',
        ]
        
        class wrapper(object):           
            def __init__(self, db):
                self.db = db 
            
            def set_theme(self, style='darkgrid', **kwargs):
                sns.set_theme(style=style, **kwargs)
                return self
                
            def __getattr__(self, name):
                
                assert name in __all__, f'"{name}" is not a valid pandas plotting method. Please choose from {__all__}'
                def foo(*args, **kwargs):
                    @parse_data(add_labels=True)
                    def boo(self, *args, **kwargs):                       
                        labels = kwargs.pop('labels', None)

                        if labels:
                            df = pd.DataFrame(dict(zip(labels, args)))
                        else:
                            df = kwargs.pop('data')
                            
                        out = getattr(sns, name)(data=df, **kwargs)  
                        # else:
                        #     raise ValueError('Keys or dataframe must be provided')                        
                        return out
                    return boo(self, *args, **kwargs)  
                
                foo.__name__ = name     
              
                return foo  
                      
        return wrapper(self.db)    
    
    @property
    def plotly(self):
        
        """
        Example:
        df = pd.DataFrame([y, yhat, time], index=['y', 'yhat', 'time']).T

        obj.plot.plotly.line(df, x='time', y=['y', 'yhat'], height=400, width=600)
        """
        
        import plotly.express as px
        import plotly.graph_objects as go
        
        __all__ =  [
                    'scatter',
                    'line',
                    'bar',
                    'histogram',
                    'box',
                    'violin',
                    'density_contour',
                    'density_heatmap',
                    'density_mapbox',
                    'density_violin',   
                    'scatter_3d',
                    'line_3d',
                    'scatter_polar',
                    'line_polar',
                    'bar_polar',
                    'scatter_ternary',
                    'line_ternary',
                    'scatter_geo',
                    'line_geo',
                    'scatter_mapbox',
                    'line_mapbox',
                    'scatter_matrix',
                    'parallel_coordinates',
                     ]
        class wrapper(object):           
            def __init__(self, db):
                self.db = db 
               
            def __getattr__(self, name):
                
                assert name in __all__, f'"{name}" is not a valid pandas plotting method. Please choose from {__all__}'
                def foo(*args, **kwargs):
                    @parse_data(add_labels=True)
                    def boo(self, *args, **kwargs):                       
                        labels = kwargs.pop('labels', None)

                        if labels:
                            df = pd.DataFrame(dict(zip(labels, args)))
                            fig = getattr(px, name)(df, **kwargs)  

                        else:
                            raise ValueError('Keys or dataframe must be provided')                        
                        return fig
                    return boo(self, *args, **kwargs)  
                
                foo.__name__ = name     
              
                return foo  
                      
        return wrapper(self.db) 
    
    @property
    def bokeh(self):
        """
        Homepage:
        https://docs.bokeh.org/en/2.4.1/index.html
        Notebooks:
        https://github.com/bokeh/bokeh-notebooks.git
        Demos:
        https://demo.bokeh.org/
        
        Example:
        fig = obj.plot.bokeh(theme='dark_minimal', output_nb=True)
        fig.figure(height=400, width=600, x_axis_label='x')
        fig.line('time', keys[2], legend_label=keys[2], color='red')
        fig.line('time', keys[3], legend_label=keys[3])
        fig.scatter('time', keys[0], line_color='yellow', legend_label=keys[0])
        fig.p.legend.location = "bottom_left"
        fig.p.legend.label_text_font_size = '8pt'
        fig.p.legend.click_policy = "hide"

        fig.show()
        
        Post html to notebook:
        from IPython.core.display import HTML
        HTML('test.html')
        
        """

        import bokeh.plotting as bp
        import bokeh.io as bio
        import bokeh.layouts as bl
                
        class wrapper(object):         
              
            def __init__(self, db):
                self.db = db 
                
            def __call__(self, **kwargs):
                                    
                if kwargs.pop('output_nb', False):
                    from bokeh.io import output_notebook
                    output_notebook()
                if kwargs.get('theme', None):
                    from bokeh.io import curdoc
                    curdoc().theme = kwargs.pop('theme')
                return self
           
            def figure(self, **kwargs):
                
                self.ax = bp.figure(**kwargs)
                return self
            
            def show(self, layout=None):
                if layout is None:
                    bp.show(self.ax)
                else:
                    bio.show(layout)
                    
                return self
            
            def layout(self, *axs, layout):

                
                axs = [ax.ax for ax in axs]
                
                return getattr(bl, layout)(*axs)
                
            
            def save(self, path):
                from bokeh.plotting import output_file, save
                
                output_file(filename=path, title="Static HTML file")
                save(self.ax)
                return self
                          
            def __getattr__(self, name):
                
                def foo(*args, **kwargs):                   
                    @parse_data(add_labels=True)
                    def boo(self, *args, **kwargs): 
                                              
                        labels = kwargs.pop('labels', None)  
                        if not isinstance(self.ax, bp.Figure):
                            self.figure()
                            
                        obj = getattr(self.ax, name)(*args, **kwargs)
                                              
                        return self
                    
                    return boo(self, *args, **kwargs)  
                
                foo.__name__ = name     
              
                return foo  
                      
        return wrapper(self.db)         
                              
    
    def set_params(self, **kwargs):
        """
        Set the figure parameters for the plots.
        
        Parameters:
        - **kwargs: The keyword arguments to be passed to the matplotlib figure function.
        """
        global figparams
        figparams = kwargs
        return self
    
    def set_axisfunc(self, func):
        """
        Set the function to be used for formatting the axes of the plots.
        
        Parameters:
        - func: The function to be used for formatting the axes of the plots.
        """
        assert callable(func), 'func must be a callable function'
        global axisfunc
        axisfunc = func
        return self
    
    @define_ax
    def gca(self, ax=None, setax=False, **kwargs):
        """
        Get the axis for plotting.
        
        Parameters:
        - ax: The matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        
        Returns:
        - ax: The matplotlib Axes object to plot on.
        """
        self.set_params(**kwargs)
        return ax
        
    @define_ax   
    @parse_data(add_labels=True)
    def histogram(self, *keys, ax=None, bins=None, **kwargs):
        """
        Generate a histogram plot.
        
        Parameters:
        - keys: The column names of the data to plot. If keys are strings, the corresponding columns will be used. 
                If keys are arrays, the arrays will be used directly.
        - ax: The matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        - bins: The number of bins to use in the histogram. If None, a default of 10 bins will be used.
                If bins is an int, it specifies the number of equal-width bins in the given range.
                If bins is a sequence, it defines the bin edges, including the rightmost edge.
        - **kwargs: Additional keyword arguments to be passed to the plot_histogram function.
        
        Returns:
        - ax: The matplotlib Axes object containing the histogram plot.
        """
        if all(isinstance(key, str) for key in keys):
            data = self.db[keys].dropna().to_numpy().T
        else:
            data = keys

        
        if ax is None:           
            _, ax = plt.subplots()
            
            
        if bins is None:
            bins = get_histo_bins(data)
            
        elif isinstance(bins, (abc.Sequence, np.ndarray)):
            bins = np.asarray(bins)
        
        elif isinstance(bins, int):
            bins = np.linspace(np.min(data), np.max(data), bins) 
              
        else:
            raise ValueError('bins must be an int or a sequence of values')

        
        plot_histogram(ax, keys, bins, **kwargs)
               
        return ax
    
    @define_ax
    @parse_data('x', 'y', 'c', 's', add_labels=True, label_keys=['x', 'y'])
    def scatter(self, x, y, ax=None, **kwargs):
        """
        Create a scatter plot.

        Parameters:
        - keys: The data to be plotted. Must contain 2-4 elements.
        - ax: The matplotlib Axes object to plot on. If not provided, a new figure and axes will be created.
        - setax: A boolean indicating whether to set the x and y axis labels. Default is True.
        - labels: A list of labels for the x and y axis. If provided, the x and y axis labels will be set accordingly.
        - kwargs: Additional keyword arguments to be passed to the scatter function.

        Returns:
        - ax: The matplotlib Axes object containing the scatter plot.
        """
        labels = kwargs.pop('labels', None) 


        ax.scatter(x, y, **kwargs)

        
        if labels and len(labels) == 2:
            ax.set_xlabel(labels[0])   
            ax.set_ylabel(labels[1])
        
        return ax
    
    @define_ax
    @parse_data('x', add_labels=True)
    def line(self, *keys, ax=None, **kwargs):
        """
        Plot a line graph.

        Parameters:
        - keys: Tuple of two elements representing the x and y values.
        - ax: Optional matplotlib Axes object to plot on.
        - labels: Optional tuple of x and y axis labels.

        Returns:
        - ax: The matplotlib Axes object with the line graph plotted.
        """
        #assert len(keys) > 0, 'keys must contain at least one element'
        
        labels = kwargs.pop('labels', None)
        x = kwargs.pop('x', None)
        
        func = lambda x, y, label=None: (ax.plot(x, y, label=label, **kwargs) if x is not None 
                                                else ax.plot(y, label=label, **kwargs))
         
        if len(keys) == 1:
            if labels is not None:
                func(x, keys[0], labels[0])
            else:   
                func(x, keys[0])
            
            if labels:
                ax.set_ylabel(labels[0]) 
       
        elif len(keys) > 1 :
            for idx, key in enumerate(keys):
                if labels is not None:
                    func(x, key, labels[idx])
                else:   
                    func(x, key)       
        return ax
     
    @define_ax
    @parse_data(add_labels=True)
    def dendrogram(self, *keys, ax=None, **kwargs):
        """
        Generate a dendrogram plot.

        Parameters:
            *keys: Variable length argument list of keys.
            ax: Optional matplotlib Axes object to plot on.
            **kwargs: Additional keyword arguments to pass to the plot_dendrogram function.

        Returns:
            ax: The matplotlib Axes object containing the dendrogram plot.
        """
        labels = kwargs.pop('labels')
        df = pd.DataFrame(dict(zip(labels, keys)))        
        plot_dendrogram(ax, df.dropna(), **kwargs)       
        return ax
    
    @define_ax
    @parse_data(add_labels=True)
    def parallel_coordinate(self, *keys, ax=None, **kwargs):
        """
        Plots a parallel coordinate plot based on the given keys.

        Parameters:
            *keys: The keys used to create the parallel coordinate plot.
            ax (optional): The matplotlib Axes object to plot on.
            **kwargs: Additional keyword arguments to customize the plot.

        Returns:
            The matplotlib Axes object containing the parallel coordinate plot.
        """
        labels = kwargs.pop('labels')
        df = pd.DataFrame(dict(zip(labels, keys)))            
        default = dict(facecolor='none', lw=0.3, alpha=0.5, edgecolor='g')
        default.update(kwargs)        
        plot_parallel_coordinate(host=ax, df=df.dropna(), **default)
        
        return ax
    
    @define_ax
    @parse_data(add_labels=True)
    def parallel_coordis(self, *keys, **kwargs):
        """
        Generate a parallel coordinates plot using the given keys as dimensions.

        Parameters:
        - keys: The keys to be used as dimensions for the parallel coordinates plot.
        - kwargs: Additional keyword arguments to be passed to the `parallel_coordis` function.

        Returns:
        - self: The current instance of the class.

        Example usage:
        ```
        labels = ['A', 'B', 'C']
        keys = [1, 2, 3]
        parallel_coordis(labels, *keys, color='blue')
        ```
        """
        labels = kwargs.pop('labels')
        df = pd.DataFrame(dict(zip(labels, keys)))               
        parallel_coordis(df.values.T, **kwargs)
        
        return self
    
    @define_ax
    def heatmap(self, *keys, ax=None, **kwargs):
        """
        Generate a heatmap plot based on the correlation between the specified keys.

        Parameters:
            *keys: Variable length argument list of keys to calculate correlation.
            ax: Optional matplotlib Axes object to plot the heatmap on.
            **kwargs: Additional keyword arguments to customize the plot.

        Returns:
            ax: The matplotlib Axes object containing the heatmap plot.
        """
        import seaborn as sns

        cmap = kwargs.pop('cmap', sns.diverging_palette(230, 20, as_cmap=True))

        corr = self.db.signals.corr(*keys)

        default = dict(annot=True, cmap=cmap, cbar=False, vmin=-1, vmax=1)

        default.update(kwargs)

        sns.heatmap(corr, ax=ax, **default)
        ax.set_aspect('equal')

        return ax
    
    @define_ax
    def wordcloud(self, text=None, ax=None, remove=('\n', '\t'), **kwargs):
        """
        Generate a word cloud visualization.

        Parameters:
            text (str): The text to generate the word cloud from. If not provided, it will use the comments from the data provider.
            ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot the word cloud on. If not provided, a new figure will be created.
            **kwargs: Additional keyword arguments to customize the word cloud.

        Returns:
            matplotlib.axes.Axes: The matplotlib axes containing the word cloud plot.
        """

        from datasurfer.lib_plots.plot_collection import plot_wordcloud

        if text is None and hasattr(self.db, 'comments'):
            text = ' '.join([txt for txt in self.db.comments().values.tolist() if isinstance(txt, str)])
        else:
            raise ValueError("Expect text input for word cloud.")

        for r in remove:
            text = text.replace(r, ' ')
            
        ax = plot_wordcloud(ax=ax, text=text, **kwargs)

        return ax
    
    @define_ax
    @parse_data(add_labels=True)
    def kde(self, *keys, ax=None, **kwargs):
        """
        Plot the kernel density estimate (KDE) for the given keys.

        Parameters:
        *keys : array-like
            The keys for which the KDE will be computed and plotted.
        ax : matplotlib.axes.Axes, optional
            The axes on which the KDE plot will be drawn. If not provided, a new figure and axes will be created.
        **kwargs : dict
            Additional keyword arguments to customize the KDE plot.

        Returns:
        ax : matplotlib.axes.Axes
            The axes on which the KDE plot is drawn.

        """
        from datasurfer.lib_signals.distrib_methods import get_kde

        lbls = kwargs.pop('labels', [None]*len(keys))
        num = kwargs.pop('count', 100)
        pltkws = kwargs.pop('plot_kws', {})

        for key, lbl in zip(keys, lbls):
            kde = get_kde(key, **kwargs)
            x = np.linspace(np.min(key), np.max(key), num)
            ax.plot(x, kde(x), label=lbl, **pltkws)

        return ax
        
    
    
       
if __name__ == '__main__':
    
    pass