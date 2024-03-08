# -*- coding: utf-8 -*-

import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri 

from matplotlib.collections import LineCollection

#%%---------------------------------------------------------------------------#  
def find_enclosed_data(x, y, linedata, boundary=True):
    
    from scipy.spatial import Delaunay
    
    if not np.allclose(linedata[0], linedata[-1]):
        
        linedata = np.vstack([linedata, linedata[0]])
    
    xy = np.vstack((x, y)).T
    
    hull = Delaunay(linedata)
    
    res = hull.find_simplex(xy)
    
    mask= res >= 0 if boundary else res > 0
    
    return mask


#%%---------------------------------------------------------------------------#  
    
def trigrid(x, y, axis=None, levels=None):
    
    '''
        generate structured mesh as triangle-object for triplot-useage
        
        arguments:
            
            x, y: x, y-coordinates
            axis: None, 0 or 1 as sorting-key, e.g. 0 means sorting using x-values.
            
        returns:
            
            object of matplotlib.tri.Triangulation
    '''
    
    def get_tris():
        
        for ilvl in range(1, len(levels)):
            
            xymask = (axarr >= levels[ilvl-1]) & (axarr <= levels[ilvl])

            order = Delaunay(xy[xymask]).simplices    
            
            yield xyidx[xymask][order]

    
    from scipy.spatial import Delaunay
    
    
    if levels is None:
        xlevels = np.unique(x)
        ylevels = np.unique(y)
    else:
        xlevels = ylevels = levels
    
    if axis is None:
        
        if xlevels.size <= ylevels.size:
            
            axis = 0
        else:
            
            axis = 1
    
    assert axis in (0, 1)
    
    levels, axarr = (xlevels, x) if axis == 0 else (ylevels, y)
    
    xy = np.vstack([x, y]).T
    xyidx = np.arange(len(xy))

    triangles = np.vstack(get_tris())
    
    return tri.Triangulation(x, y, triangles=triangles)
    
    

#%%---------------------------------------------------------------------------#  
def gentris(x, y, k=1.5):
    
    '''generate triangles for given coordinates (x, y)
       k: factor 0 < k < inf, to filter not convex triangles
    '''
    
    normalize = lambda x: (x - x.min()) / x.ptp()

    xx = normalize(x)
    yy = normalize(y)
    
    dxx = np.diff(np.sort(xx)).max()
    dyy = np.diff(np.sort(yy)).max()
    
    maxdis = np.sqrt(dxx**2 + dyy**2)
    
    tris = tri.Triangulation(xx, yy)
    tarr = tris.triangles
    
    xarr = np.hstack([xx[tarr], xx[tarr][:, 0].reshape(-1, 1)])
    yarr = np.hstack([yy[tarr], yy[tarr][:, 0].reshape(-1, 1)])
    
    xy = np.swapaxes(np.dstack([xarr, yarr]).T, 0, 1) # (4, 2, len)
    
    
    dis = np.sqrt(np.vstack((np.diff(xy[n:n+2], axis=0)**2).sum(axis=1) for n in range(3)))

    
    return tri.Triangulation(x, y, triangles=tarr[np.all(dis <= maxdis*k, axis=0)])


#%%---------------------------------------------------------------------------#  
def interp_contour(x, y, num, find_nearst=0.5):
    
    ''' 
        linear interpolate contour line (x, y) to size "num"
        find_nearst: 0 <= x <= 0.5, if x == 0, will not use the nearst original data.
                     
    '''
    
    from scipy.interpolate import interp1d
    
    x_linear = np.arange(x.size)
    
    
    new_xlin = np.linspace(0, x_linear[-1], num)

    if find_nearst and find_nearst > 0:
        
        assert find_nearst <= 0.5
        
        dxlin = new_xlin[1]
        dist = np.abs((np.atleast_2d(x_linear) - np.atleast_2d(new_xlin).T))

        
        mask = dist.min(axis=1) < find_nearst * dxlin
        
        new_xlin[mask] = np.round(new_xlin[mask])
    
        
    dx = np.hstack([np.diff(x), 0])
        
    xbase_idx = np.floor(new_xlin).astype(int)
    
    dx_factor = new_xlin - xbase_idx
    
    xbase = x[xbase_idx]
    
    new_x = xbase + dx[xbase_idx] * dx_factor
    
    new_y = interp1d(x_linear, y, assume_sorted=True)(new_xlin)
    
    return new_x, new_y

#%%---------------------------------------------------------------------------#  

def tricontourdata(*args, **kwargs):
      
    import matplotlib._tri as _tri    
    from matplotlib.tri.tricontour import TriContourSet
    
    def get_segs(levels):
        
        for level in levels:

            array = cppContourGenerator.create_contour(level)
            
            if array:
            
                yield level, array[0]
                
            else:
                
                yield level, np.atleast_2d(np.nan * np.ones(2))
    
    def get_closed_segs(levels):
        
        if levels.size == 1:
            
            levels = np.hstack([levels, np.inf])

        
        for idx, level in enumerate(levels[:-1]):
            
            rg = (levels[idx], levels[idx+1])
            array = cppContourGenerator.create_filled_contour(*rg)[0]
            
            if array.size:
                
                if not np.allclose(array[0], array[-1]):
                    
                    array = np.vstack([array, array[0]])
                                                    
                
                yield rg, array
            else:
                yield rg, np.atleast_2d(np.nan * np.ones(2))
                
    def interp(segs, size):
        
        if not size:
            
            return segs
        
        else:
            
            func = (lambda seg: np.nan * np.ones((size, 2)) if
                                np.all(np.isnan(seg)) else
                                np.vstack(interp_contour(*seg.T, size)).T)
            return np.vstack(func(seg)[np.newaxis, :, :] for seg in segs)
    
    size = kwargs.pop('size', None)
    
    if isinstance(args[0], TriContourSet):
        
        cs, = args
        
        segs = [(lambda s: s[0] if s[0].size else 
                 np.atleast_2d(np.nan * np.ones(2)))(sgs) for 
                 sgs in cs.allsegs]
        levels = cs.levels
               
    elif isinstance(args[0], tri.Triangulation):
                
        tris, z, *_args = args
        
        cppContourGenerator = _tri.TriContourGenerator(tris.get_cpp_triangulation(), z)
        
        if _args:
            
            levels = np.array(_args).ravel()
            
        else:
            
            levels = np.asarray(kwargs.pop('levels')).ravel()
                        
        if kwargs.pop('closed',  False):
            
            levels, segs = zip(*get_closed_segs(levels))
                    
        else:
            
            levels, segs = zip(*get_segs(levels))
            
#        if _args and len(levels) == 1:
#            
#            return interp(segs, size)[0].T
        
    else:
        
        x, y, *_args = args
        
        k = kwargs.pop('k', 1.5)
        
        kwargs['size'] = size
        
        return tricontourdata(gentris(x, y, k=k), *_args, **kwargs)

        
    return levels, interp(segs, size)

#%%---------------------------------------------------------------------------# 
def labeledline(line, pos=None, **kwargs):

        
    def get_txt(pos):
        
        posnom = (pos - xymin) / xyptp

    
        idx = ((xynom - np.asarray(posnom, dtype=float))**2).sum(axis=1).argmin()
        pos = xydata[idx]
                
        ang = np.arctan2(*np.diff(xydata, axis=0)[min(idx, len(xydata)-2)][::-1]) * 180.0 / np.pi
        
        trans_angle = ax.transData.transform_angles(np.array((ang,)), np.atleast_2d(pos))[0]
        
        if trans_angle > 90:
            trans_angle = trans_angle - 180.0
        if trans_angle < -90:
            trans_angle = 180.0 + trans_angle
        
        
        pixel_dat = ax.transData.transform(xydata)
        
        pixel_pos = ax.transData.transform(np.atleast_2d(pos))

        
        settings = dict(clip_on=True, 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        color=line.get_color(),
        #                    backgroundcolor=ax.get_facecolor(),
                        )
        
        settings.update(kwargs)
        
        txt = ax.text(*pos, label, rotation=trans_angle, 
        #                  bbox=dict(facecolor='none', edgecolor='none', pad=3.0),
                      **settings)
        
        #    bbox = txt.get_bbox_patch().get_bbox()
        
        
        if not 'backgroundcolor' in settings:
        
            bb = txt.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
            
            bbradius = space + bb.width/2 #np.sqrt((bb.width**2 + bb.height**2)) / 2 
            
            line.set_xdata(np.nan)
            
            mask = ((pixel_dat - pixel_pos)**2).sum(axis=1) <= bbradius**2
            
            xydata[mask] = np.nan
            
            xdata, ydata = xydata.T
            
            line.set_xdata(xdata)
            
            line.set_ydata(ydata)
            
        return txt
    
    
    ax = line.axes
            
    label = kwargs.pop('label', line.get_label())
    space = kwargs.pop('space', 0)  
    
    xydata = np.vstack((line.get_xdata(), line.get_ydata())).T
    
    xymin = xydata.min(axis=0)
    
    xyptp = xydata.ptp(axis=0)
    
    xynom = (xydata - xymin) / xyptp

    
    if pos is None:        
        pos = xydata.mean(axis=0)
        
    pos = np.atleast_2d(pos)
    
    return list(map(get_txt, pos))

#%%---------------------------------------------------------------------------# 
def labeledlines(lines, pos=None, **kwargs):

    return [labeledline(line, pos, **kwargs) for line in lines]    

#%%---------------------------------------------------------------------------#  
def register_radar(array=None, **kwargs):
    
    from matplotlib.path import Path
    from matplotlib.spines import Spine
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection


    def unit_poly_verts(theta):
        """Return vertices of polygon for subplot axes.
    
        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
        """
        x0, y0, r = [0.5] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts
    
    def draw_poly_patch(self):
        verts = unit_poly_verts(self.theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)
    
    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
     
    titles = kwargs.pop('titles', None)
    
    if titles and array is None:
        array = len(titles)

    frame = 'circle' if array is None else kwargs.pop('frame', 'polygon') 

    class RadarAxes(PolarAxes):
    
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1


        def __init__(self, *args, **kwargs):
            
            if array is None:
                
                self.theta = None
                self.xydata = np.nan
            else:
            
                if isinstance(array, (int, float)):
                    
                    self.xydata = np.nan * np.ones((1, int(array)))
                else:
                    self.xydata = np.atleast_2d(np.asarray(array))
                
                num, dim = self.xydata.shape
                # calculate evenly-spaced axis angles
                self.theta = np.linspace(0, 2*np.pi, dim, endpoint=False)
                # rotate theta such that the first axis is at the top
                self.theta += np.pi/2  
                
            self.titles = titles

            
            super().__init__(*args, **kwargs)
        
    
        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)
    
        def plot(self, *args, **kwargs):
            
            if args and not isinstance(args[0], str):
                                    
                array, *args = args
                
                array = np.atleast_2d(np.asarray(array))
                
                if np.all(np.isnan(self.xydata)):
                    
                    self.xydata = array
                else:
                    self.xydata = np.vstack([self.xydata, array])

            else:
                
                assert not np.all(np.isnan(self.xydata))
                array = self.xydata
            
            num, dim = self.xydata.shape
            
            if self.theta is None or self.theta.size != dim:
                self.theta = np.linspace(0, 2*np.pi, dim, endpoint=False)
                # rotate theta such that the first axis is at the top
                self.theta += np.pi/2  
                       
            titles = kwargs.pop('titles', None)
            
            if titles is not None:
                
                self.titles = titles
            labels = kwargs.pop('labels', [None] * num)
            rticks = kwargs.pop('rticks', [])
            colors = kwargs.pop('colors', [None] * num)            
            alpha = kwargs.pop('alpha', 0.2) 
            filled = kwargs.pop('filled', True)
            
            lines = []
            
            for n, arr in enumerate(array):
                
                _kwargs = dict(label=labels[n])
                _kwargs.update(kwargs)
                
                if colors[n]:   
                    line, = super().plot(self.theta, arr, *args, color=colors[n], **_kwargs)
                else:
                    line, = super().plot(self.theta, arr, *args, **_kwargs)
                    
                self._close_line(line)
                
                if filled:
                    self.fill(self.theta, arr, facecolor=line.get_color(), alpha=alpha)
                
                lines.append(line)

            self.set_varlabels(self.titles)
            
            self.set_rlabel_position(90)
                      
            self.set_yticklabels(rticks, fontsize=8, color='b')
            
            for tick, d in zip(self.xaxis.get_ticklabels(), self.theta):
                
                if np.pi / 2 < d < np.pi * 1.5:
                    tick.set_ha('right')
                elif np.pi * 1.5 < d:
                    tick.set_ha('left')
                    
                tick.set_color('black')
                
            return lines
    
        def _close_line(self, line):
            x, y = line.get_data()

            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
    
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(self.theta), labels)
    
        def _gen_axes_patch(self):
            return patch_dict[frame](self)
    
        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.
    
            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(self.theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)
    
            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}
        
    register_projection(RadarAxes)
    
    return RadarAxes.name


#%%---------------------------------------------------------------------------#
def radar_chart(array, *args, **kwargs):
    
    from matplotlib.path import Path
    from matplotlib.spines import Spine
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection


    def unit_poly_verts(theta):
        """Return vertices of polygon for subplot axes.
    
        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
        """
        x0, y0, r = [0.5] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts
    
    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)
    
    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    frame = kwargs.pop('frame', 'polygon')    
    num, dim = np.asarray(array).shape
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, dim, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2


    class RadarAxes(PolarAxes):
    
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]
    
    
        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)
    
        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
                
            return lines
    
        def _close_line(self, line):
            x, y = line.get_data()

            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
    
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
    
        def _gen_axes_patch(self):
            return self.draw_patch()
    
        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.
    
            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)
    
            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    
    titles = kwargs.pop('titles', ['Input {}'.format(n) for n in range(dim)])
    labels = kwargs.pop('labels', [None] * num)
    rticks = kwargs.pop('rticks', [])
    colors = kwargs.pop('colors', [None] * num)
    
    alpha = kwargs.pop('alpha', 0.2)
    
    if not args:        
        args = (111, )
        
    ax = plt.gcf().add_subplot(*args, projection='radar', **kwargs)
    
    for n, arr in enumerate(array):
        
        line, = ax.plot(theta, arr, color=colors[n], label=labels[n])

        ax.fill(theta, arr, facecolor=line.get_color(), alpha=alpha)

    ax.set_varlabels(titles)
    
    ax.set_rlabel_position(90)
              
    ax.set_yticklabels(rticks, fontsize=8, color='b')

    for tick, d in zip(ax.xaxis.get_ticklabels(), theta):
        
        if np.pi / 2 < d < np.pi * 1.5:
            tick.set_ha('right')
        elif np.pi * 1.5 < d:
            tick.set_ha('left')
            
        tick.set_color('black')
    
    return ax


#%%---------------------------------------------------------------------------#
def reg_radar(titles, **title_opts):

    from matplotlib.path import Path
    from matplotlib.spines import Spine
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection


    def unit_poly_verts(theta):
        """Return vertices of polygon for subplot axes.
    
        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
        """
        x0, y0, r = [0.5] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts

    def draw_poly_patch(self):
        
        verts = unit_poly_verts(THETA)
        
        return plt.Polygon(verts, closed=True, edgecolor='k')
    
    TITLES = list(titles)
    
    THETA = np.linspace(0, 2*np.pi, len(TITLES), endpoint=False) + np.pi/2
    
    class RadarAxes(PolarAxes):
    
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            
            self.dim = len(TITLES)
            
            self.theta = np.linspace(0, 2*np.pi, self.dim+1) + np.pi/2
            
            super().__init__(*args, **kwargs)
                        
            super().grid(False)

            self.rgrid() 
            
            title_opts_ = dict(fontsize=12, weight="bold", color="black")
            
            title_opts_.update(title_opts)
            
            
            self.set_thetagrids(np.degrees(THETA).astype(np.float32), 
                                labels=TITLES, **title_opts_)
            
            self.set_xlim(0, np.pi * 2)
            self.set_ylim(0, 1)

            self.set_yticklabels([])
            for tick, d in zip(self.xaxis.get_ticklabels(), THETA):
                
                if np.pi / 2 < d < np.pi * 1.5:
                    tick.set_ha('right')
                elif np.pi * 1.5 < d:
                    tick.set_ha('left')

        def _gen_axes_patch(self):
            
            return draw_poly_patch(self)
        
    
        def _gen_axes_spines(self):

            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.
    
            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(THETA)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)
    
            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            
            return {'polar': spine}
            
        def rgrid(self, b=True, count=5, **kwargs):
            
            if hasattr(self, '_rgrids'):
                for col in self._rgrids:
                    
                    col.remove()
            
            if b:
            
                defaults = dict(color='grey', lw=0.5)       
                defaults.update(kwargs)
                
                dy = 1 / count
                            
                ys = np.ones_like(self.theta) * np.arange(dy, 1+dy, dy).reshape(-1, 1)
                
                xs = np.tile(self.theta, (count, 1))
                
                xys = np.dstack([xs, ys])
        
                line_segments = LineCollection(xys, **defaults)
                
                line_colls1 = self.add_collection(line_segments)
                
                xs = np.tile(THETA.reshape(-1, 1), 2)
                ys = np.tile([0, 1], (self.dim, 1))
                
                xys = np.dstack([xs, ys])
                line_segments = LineCollection(xys, **defaults)
                
                line_colls2 = self.add_collection(line_segments)
                                
                self._rgrids = (line_colls1, line_colls2)

        def set_zebra(self, **kwargs):
            
            assert hasattr(self, '_rgrids')
            
            defaults = dict(color='grey', alpha=0.2)
            
            defaults.update(kwargs)
            arr = np.dstack(p.vertices for p in self._rgrids[0].get_paths())
            self._rgrids[0].set_visible(defaults.pop('edge', False))
            xs, ys = arr[:, 0, :], arr[:, 1, :]
            
            xs = xs.T
            ys = ys.T
            

            return [self.fill_between(xs[2*i], ys[2*i], ys[2*i+1], 
                        zorder=0, **defaults) for i in range(len(xs) // 2)]
            
        def get_theta(self, title):
            
            return THETA[TITLES.index(title)]
        
        def scale(self, array, *arg,  **kwargs):
            
            assert hasattr(self, '_rgrids')

            origin = kwargs.pop('include_origin', False)
            apply = kwargs.pop('apply_tick', False)
            kind = kwargs.pop('kind', 'max')
            
            arr = np.dstack(p.vertices for p in self._rgrids[0].get_paths())

            _, _, n = arr.shape
            

            arrays, ticks = scale_radar(array, n=n+1, kind=kind)
            
            if not origin:
                
                ticks = ticks[:, 1:]
            
            if apply:
                
                for title, labels in zip(TITLES, ticks):
                    
                    self.set_rlabel(title, labels, include_origin=origin, **kwargs)
            
            return arrays, ticks
            
        def set_rlabel(self, title, labels, **kwargs):
            
            def get_txt():
                
                for xy, lbl in zip(xys, labels):
                    
                    yield self.text(*xy, fmt(lbl), **kwargs)
                    
            assert hasattr(self, '_rgrids')
            assert hasattr(labels, '__iter__')
            
            fmt = kwargs.pop('fmt', str)
            origin = kwargs.pop('include_origin', False)
            
            array = np.dstack(p.vertices for p in self._rgrids[0].get_paths())
            

            idx = TITLES.index(title)
            xys = array[idx].T
            if origin:

                xys = np.concatenate([[[0, 0]], xys], axis=0)            
            
            return list(get_txt())
        
        def plot(self, array, *args, **kwargs):
            
            clustered = kwargs.pop('clustered', False)            
            yarray = np.atleast_2d(np.asarray(array))
            
            assert np.all((yarray >= 0) & (yarray <= 1))
            num, dim = yarray.shape
            
            if dim != self.dim and num == self.dim:
                
                yarray = yarray.T
            
                num, dim = dim, num
            
            assert dim == self.dim
            
            yarray = np.concatenate([yarray, yarray[:, :1]], axis=1)
            
            if clustered:
                
                xys = np.dstack([np.tile(self.theta, (num, 1)), yarray])
                line_segments = LineCollection(xys, *args, **kwargs)
                
                lines = self.add_collection(line_segments)
            
            else:
                lines = super().plot(self.theta, yarray.T, *args, **kwargs)
            
            return lines
        
        def fill_lines(self, lines, **kwargs):
            
            colors = kwargs.pop('colors', [None]*len(lines))
            
            assert len(colors) == len(colors)
            
            for line, c in zip(lines, colors):
                
                color = line.get_color() if c is None else c
                super().fill(*line.get_xydata().T, color=color, **kwargs)
                
            return self
        
    register_projection(RadarAxes)
    
    return RadarAxes.name 
#%%---------------------------------------------------------------------------#
def scale_radar(array, n=6, kind='max'):
    
    from matplotlib.ticker import MaxNLocator, LinearLocator
    
    
    if array.ndim == 1:
        
        array = np.atleast_2d(array).T
        
    if kind == 'linear':
        locator = LinearLocator(n)
    else:
        locator = MaxNLocator(n)
        
    tvalues = ([locator.tick_values(*aminmax) for aminmax
                        in zip(array.min(axis=0), array.max(axis=0))])
        
    aminmax = np.vstack((arr.min(), arr.max()) for arr in tvalues)
    amins = aminmax[:, 0]
    aptps = aminmax.ptp(axis=1)

    
    ticks = np.vstack(tvalue[:n] for tvalue in tvalues)
    
    arrays = (array - amins) / aptps
    
    return arrays, ticks

#%%---------------------------------------------------------------------------#
def parallel_coordinates(array, **kwargs):
    
    def init_axes():
        
        for i, pos in enumerate(xpositions):
        
            axes = ax.twinx()
            
            axes.spines['top'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)
            
            axes.spines['right'].set_color('black')
            axes.tick_params(axis='y', colors='black')
            
            axes.set_ylim((0, 1))
            
            axes.yaxis.set_label_position("left")
            axes.spines['right'].set_position(('axes', pos))
            axes.spines['right'].set_linewidth(1)
            axes.yaxis.set_tick_params(width=1, length=15)
            
            tick_ = ntick if arr_valid[i] else 3
            axes.set_yticks(np.linspace(0, 1, tick_))
            axes.set_yticklabels(['{:0.2E}'.format(s) for s in 
                                  np.linspace(arr_min[i], arr_max[i], tick_)], fontsize=8)
    
    ax = kwargs.pop('ax', plt.gcf().gca())
        
    num, dim = np.asarray(array).shape
        
    titles = kwargs.pop('titles', ['Input {}'.format(n) for n in range(dim)])
        
    ntick = kwargs.pop('ntick', 11) 
    
    xpositions = np.linspace(0, 1, dim)
    
    arr_max = kwargs.pop('amax', array.max(axis=0))
    arr_min = kwargs.pop('amin', array.min(axis=0))
    
    arr_valid = arr_max > arr_min

    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('axes', -0.03))
    ax.spines['left'].set_visible(False)
    
    ax.set_xticks(xpositions)
    ax.set_xticklabels(titles, minor=False, 
                       **kwargs.pop('opt_xlabel', 
                                    dict(rotation=-45, ha='left', fontsize=10)))
    
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))   
    
    init_axes()
    
    yarrays = (array - arr_min) / (arr_max - arr_min)
    yarrays[:, ~arr_valid] = 0.5
    
    default = dict(linewidth=0.8, color='grey', alpha=0.5)
    default.update(kwargs)
    lines = ax.plot(xpositions, yarrays.T, **default) 


    for tick in ax.xaxis.get_ticklabels():
        tick.set_color('black')
    
    return ax, lines, xpositions, yarrays


#%%---------------------------------------------------------------------------#
def parallel_coordis(array, **kwargs):
    
    from matplotlib.ticker import AutoMinorLocator
    
    def get_paralax():
        
        for i, y in enumerate(array.T):
            
            x = np.zeros_like(y)
            axes = ax.twinx()
            axes.spines['top'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)
            
            axes.spines['right'].set_color('black')
            axes.tick_params(axis='y', colors='black')        
            axes.yaxis.set_label_position("left")
            axes.spines['right'].set_position(('axes', xlocs[i]))
            axes.spines['right'].set_linewidth(1)
            axes.yaxis.set_tick_params(width=1, length=15, which='major')
            axes.yaxis.set_tick_params(width=1, length=4, which='minor')
            
            axes.plot(x, y, alpha=0)
            axes.yaxis.set_minor_locator(AutoMinorLocator())
            axes.set_ylim(ylims[i])
            
            yield axes, axes.transData.transform(np.vstack([x, y]).T)[:, 1]
    
    num, dim = np.asarray(array).shape
        
    ax = kwargs.pop('ax', plt.gcf().gca())        
    titles = kwargs.pop('titles', ['Parameter {}'.format(n) for n in range(dim)])  
    title_opts_ = kwargs.pop('title_opts', {})
    line_opts = kwargs.pop('line_opts', dict())
    ylims = kwargs.pop('ylims', [[None] * 2] * dim)
    clustered = kwargs.pop('clustered', True) 
    title_top = kwargs.pop('title_top', False)
    xlocs = kwargs.pop('xlocs', None)
    violin = kwargs.pop('violin', False)
    
    mask = np.asarray(kwargs.pop('mask', np.ones(dim)), dtype=bool)
    

    if violin and xlocs is None:
        xlocs = ((np.arange(0, 1, 1/(dim)) 
                +np.arange(1/(dim), 1+1/(dim), 1/(dim)))/2) 
    else:
        
        if xlocs is None:
            xlocs = np.linspace(0, 1, dim)
            
        xlocs = np.sort(np.asarray(xlocs))
        
        if not np.all((xlocs>=0) & (xlocs<=1)):
            warnings.warn('Expect x locations inside of [0, 1].')

    axs = titles if all(isinstance(ax_, matplotlib.axes.Axes) for ax_ in titles) else []
    
    assert len(xlocs) == dim and len(mask) == dim
    if kwargs:
        
        raise KeyError(kwargs.keys())     
    
    if axs:
        
        assert len(axs) == dim 
        xlocs = np.asarray(ax.get_xticks())
        ys = [ax_.transData.transform(np.vstack([np.zeros_like(y), y]).T)[:, 1] 
                for ax_, y in zip(axs, array.T)]
         
    else:  
        ax.set_xlim(0, 1)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if title_top == True:
            title_opts = dict(rotation=0, ha='center', fontsize=10, weight='bold',)
            ax.xaxis.tick_top()
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(length=0)
        else:    
            title_opts = dict(rotation=-45, ha='left', fontsize=8,)        
            ax.spines['bottom'].set_position(('axes', -0.03))
        
        title_opts.update(title_opts_)    
   
        ax.set_xticks(xlocs)
        ax.set_xticklabels(titles, minor=False, **title_opts)
        
        axs, ys = zip(*get_paralax())
    
    ys = np.asarray(ys)

    xs = ax.transData.transform(np.vstack([xlocs, np.zeros_like(xlocs)]).T)[:, 0]
    inv = ax.transData.inverted()
    
    xys = np.vstack([np.tile(xs, num), ys.T.ravel()]).T
    
    xys = inv.transform(xys).reshape(num, dim, 2)
    
    ys = xys[..., 1]


    if violin:  
        line_opts_ = dict(widths=np.diff(xlocs[mask]).min(), showmeans=False, showextrema=True, showmedians=False)
        
        
        line_opts_.update(dict((k, line_opts.pop(k)) for k in line_opts_ if k in line_opts))
        lines = ax.violinplot(ys[:, mask], xlocs[mask], **line_opts_) 
        
     
        
        if 'cbars' in lines:
            lines['cbars'].set_color('none')
        
        barcolor = line_opts.pop('barcolor', 'k')
        barwidth = line_opts.pop('barwidth', 1)
        for partname in ('cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in lines:
                vp = lines[partname]
                vp.set_color(barcolor)
                vp.set_linewidth(barwidth)

            
        for pc in lines['bodies']:
            line_opts_ = dict(alpha=0.8, linewidth=0.5, edgecolor='k', facecolor=None)
            line_opts_.update(line_opts)
            pc.set(**line_opts_)

    else:

        if clustered:
                        
            line_segments = LineCollection(xys[:, mask, :], **line_opts)
            
            lines = ax.add_collection(line_segments)        
        else:                
            lines = ax.plot(xlocs, ys[:, mask].T, **line_opts)
    
    plt.sca(ax)

    return axs, lines

#%%---------------------------------------------------------------------------#
def triviolin_kde(y0body, y0, yscale, **kwargs):
    
    from scipy import stats
       
    p, = y0body.get_paths()
    
    arr = p.vertices if kwargs.pop('axis', 1) else p.vertices.T 
    
    u, w = arr.T
    
    side = kwargs.pop('side', 'both')
    
    if side == 'left':
        u = u.clip(u.min(), u.mean())
        
    elif side == 'right':
        u = u.clip(u.mean(), u.max())
       
    tris = trigrid(u+kwargs.pop('offset', 0), w)
    
    assert not kwargs.keys()

    w0, inv_idx = np.unique(w, return_inverse=True)
    
    fct = (w0-w0.min()) / w0.ptp()
    
    yscale1 = yscale.min() + yscale.ptp()*fct
    
    y1 = y0.min() + y0.ptp()*fct
    
    X, Y = [arr.T for arr in np.meshgrid(yscale1, y1)]

    positions = np.vstack([X.ravel(), Y.ravel()])
    
    try:
        kernel = stats.gaussian_kde(np.vstack([yscale, y0]))
        Z = kernel(positions).reshape(X.shape)
        
        z1 = yscale1[Z.argmax(axis=0)]
        
    except np.linalg.LinAlgError:
        
        z1 = yscale1
    
    return tris, z1[inv_idx]

#%%---------------------------------------------------------------------------#
def locate_tick(axis, base_major=None, base_minor=None):
    
    from matplotlib.ticker import MultipleLocator
    from matplotlib.ticker import AutoMinorLocator

    
    if isinstance(axis, matplotlib.axes.Axes):
        
        locate_tick(axis.xaxis, base_major, base_minor)
        locate_tick(axis.xaxis, base_major, base_minor)
        
        return axis
    
    if base_major is not None:
        
        locator_major = MultipleLocator(base_major)
        axis.set_major_locator(locator_major)
    
    if base_minor is not False:
        

        
        if base_minor is None:
            
            locator_minor = AutoMinorLocator()
        
        else:
            
            locator_minor = MultipleLocator(base_minor)
        
                   
        axis.set_minor_locator(locator_minor)
    
    return axis
    
    
#%%---------------------------------------------------------------------------#
if __name__ == '__main__':
    

    titles = [s.strip().capitalize() for s 
              in open('pareto_overview.csv', 'r').readline().strip('#').split(';')]
    
    array = np.loadtxt('pareto_overview.csv', delimiter=';')

    array = array[(array[:, 7] < 55) & (array[:, 8] < 2)]
    array = (array - array.min(axis=0)) / (array.max(axis=0) - array.min(axis=0))    
    
    
    fig = plt.figure(tight_layout=True, figsize=(14, 7), dpi=120)
    
    ax0 = fig.add_subplot(111, projection=register_radar(titles=titles[-6:]))
#    
    ax0.plot(array[:3, -6:], labels=['Design {}'.format(i) for i in range(len(array))])

#    ax.plot()
    ax0.plot(array[6, -6:], label='sum')

    ax0.grid(True, color='b', ls=':')
    
    ax0.set_ylim(0, 1)
    ax0.set_xlim(0, 2*np.pi)

    ax0.legend(loc=1).draggable(True)
    
#    ax = fig.add_subplot(212)
#    
#    parallel_coordinates(array[:20], titles=titles, color='blue', alpha=1, linewidth=0.2)
#    print('\n'.join(sorted(dir(ax.xaxis.get_ticklabels()[0]))))
    plt.show()
    
    
    
