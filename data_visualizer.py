import os
from typing import Optional, Union, Sequence, Tuple

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    # 'text.usetex': True,
    'pgf.rcfonts': False,
})


def _cycle(iterable, index, default=None):
    if len(iterable) == 1 and default is not None:
        return default
    return iterable[index % len(iterable)]


def _tensor_depth(tensor):
    if hasattr(tensor, 'shape'):
        return len(tensor.shape)
    elif hasattr(tensor, '__getitem__') and len(tensor) > 0 and not isinstance(tensor, str):
        return 1 + _tensor_depth(tensor[0])
    return 0


def _to_categorical(values, unique_values=None):
    if unique_values is None:
        unique_values = np.unique(values)
    if isinstance(unique_values, np.ndarray):
        unique_values = list(unique_values)
    return np.array([unique_values.index(_) for _ in values], dtype=np.int)


def _orthogonal(X, dim):
    if dim == 1:
        return X[:, 0], np.zeros(X.shape[0])
    elif dim == 2:
        return X[:, 0], X[:, 1]
    return X[:, 0], X[:, 1], X[:, 2]


def _group_by(Xs, values, unique_values=None):
    if unique_values is None:
        unique_values = np.unique(values)
    ndim = _tensor_depth(Xs)
    if not isinstance(Xs, np.ndarray):
        Xs = [np.array(X) for X in Xs] if ndim > 2 else np.array(Xs)
    if ndim <= 2:
        return [Xs[np.where(values == c)[0]] for c in unique_values]
    else:
        Rs = []
        for X in Xs:
            Rs.append([X[np.where(values == c)[0]] for c in unique_values])
        return Rs


# noinspection DuplicatedCode
class DataVisualizer:
    ecmaps = ['none', 'black']
    linestyles = ['-', ':', '--', '-.']
    borderoptions = [
        dict(edgecolor='none'),
        *(dict(edgecolor='black', linestyle=ls) for ls in ['-', ':', '--', '-.'])
    ]
    markers = ['o', '^', 's', '*', 'p', 'P', 'v', 'X', 'D', 'H', "2", '$...$']

    figure_params = dict(figsize=(4, 4))
    ax_params = dict(facecolor=(0.9, 0.9, 0.9))
    plot_params = dict(linestyle='--', marker='o')
    scatter_params = dict(linewidth=1, alpha=0.6)
    legend_params = dict(fancybox=True, framealpha=0.4)
    grid_params = dict(which='both', linestyle=':')

    axe3d_scale = 1.  # 1.22
    axe3d_title_offset = 1.08

    def __init__(self):
        self.pausing: bool = False
        self.fig: Optional[plt.Figure] = None
        self.axes = []

    def plot(self,
             *args,
             title=None,
             ax=None,
             **kwargs):
        ax = self._init_axe(ax=ax)
        ax.plot(*args, **{**self.plot_params, **kwargs})

        if title is not None:
            ax.set_title(title)

    def bar(self,
            data,
            err=None,
            groups=None,
            categories=None,
            width=0.2,
            group_spacing=0.05,
            cat_spacing=0.5,
            bar_label=False,
            bar_label_padding=1,
            bar_label_rotation=0,
            percentage=False,
            pallete=None,
            group_legend=False,
            xlabel=None,
            ylabel=None,
            title=None,
            ax=None,
            **kwargs,
            ):
        n_groups = data.shape[0]
        n_categories = data.shape[1]
        if err is None:
            err = [None] * n_groups
        if groups is None:
            groups = np.arange(n_groups)
        if categories is None:
            categories = np.arange(n_categories)
        offsets = np.arange(-(n_groups - 1) / 2 * (width + group_spacing),
                            (n_groups - 1) / 2 * (width + group_spacing) + np.finfo(np.float32).eps,
                            width + group_spacing)

        ax = self._init_axe(ax)

        origins = np.linspace(0,
                              ((n_groups * width + (n_groups - 1) * group_spacing) + cat_spacing) * n_categories,
                              n_categories)
        if pallete is None:
            cmap = colors.ListedColormap(seaborn.color_palette(n_colors=n_groups, as_cmap=True))
        else:
            cmap = get_cmap(pallete, n_groups)
        for gi in range(n_groups):
            bars = ax.bar(origins + offsets[gi], data[gi],
                          yerr=err[gi],
                          capsize=width * 20 if err is not None else None,
                          width=width,
                          color=cmap(gi),
                          alpha=0.8,
                          **kwargs)
            if bar_label:
                ax.bar_label(bars, labels=[f'{_:.01%}' if percentage else f'{_:.01}' for _ in data[gi]],
                             padding=bar_label_padding, rotation=bar_label_rotation, fontsize=18)
        ax.set_xlim(xmin=origins[0] - (n_groups - 1) * (width + group_spacing),
                    xmax=origins[-1] + (n_groups - 1) * (width + group_spacing))
        ax.set_xticks(origins)
        ax.set_xticklabels(categories, fontsize=16)
        ax.set_ylim(ymin=0, ymax=1.05 if percentage else None)
        if percentage:
            ax.set_yticks(np.arange(0, 1.1, .1))
            ax.set_yticklabels([f"{_}%" for _ in range(0, 101, 10)], fontsize=16)

        ax.grid(axis='y', **self.grid_params)
        if group_legend:
            ax.add_artist(self.group_legend(groups, cmap, loc='upper center', ncol=n_groups))

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=20)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=20)
        if title is not None:
            ax.set_title(title)
        return ax, origins

    def scatter(self,
                X,
                y=None,
                y_unique=None,
                z=None,
                z_unique=None,
                dim=None,
                normalize=True,
                kde=False,
                cmap=None,
                title=None,
                class_legend=True,
                grid=True,
                ax=None):
        if dim is None:
            if X.shape[1] <= 3:
                dim = X.shape[1]
            else:
                dim = 2
        n_samples = X.shape[0]

        if y is None:
            y = np.array([0 for _ in range(n_samples)])
        with_categories = z is not None
        if with_categories and not _tensor_depth(z) == 1:
            raise ValueError(f"Categories must be 1 dimensional array.")
        if y_unique is None:
            y_unique = np.unique(y)
        n_classes = len(y_unique)
        if z_unique is None:
            z_unique = np.unique(z)

        X = self.embed(X, _to_categorical(y, unique_values=y_unique), dim, normalize=normalize)
        dim = X.shape[1]
        if not 0 < dim <= 3:
            raise ValueError(f"Unable to plot {dim}-dimensional data.")
        if dim == 3 and z_unique is not None and len(z_unique) > 1:
            raise ValueError("Categories are not availabe for 3d plot.")

        ax = self._init_axe(ax=ax, dim=dim)

        X = _group_by(X, y, unique_values=y_unique)
        z = _group_by(z, y, unique_values=y_unique) if with_categories else [None for _ in X]

        if cmap is None:
            cmap = get_cmap('Spectral', n_classes)
        for i, (X_i, z_i) in enumerate(zip(X, z)):
            if z_i is not None:
                X_i = _group_by(X_i, z_i, z_unique)
                for j, X_ij in enumerate(X_i):
                    if len(X_ij) > 0:
                        ax.scatter(*_orthogonal(X_ij, dim=dim),
                                   marker=self.markers[0],
                                   color=cmap(i),
                                   s=50,
                                   **_cycle(self.borderoptions, j),
                                   **self.scatter_params)
            else:
                ax.scatter(*_orthogonal(X_i, dim=dim),
                           marker=self.markers[0],
                           color=cmap(i),
                           s=50,
                           **self.scatter_params)
        if kde:
            if dim == 1:
                inset_ax = ax.inset_axes([0.0, 0.8, 1.0, 0.2])
                for i, X_i in enumerate(X):
                    seaborn.kdeplot(np.squeeze(X_i, 1),
                                    color=cmap(i),
                                    fill=True,
                                    ax=inset_ax)
                inset_ax.set_xlim(ax.get_xlim())
                inset_ax.patch.set_alpha(0)
                inset_ax.set_facecolor(ax.get_facecolor())
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                inset_ax.set_xlabel('KDE')
                inset_ax.set_ylabel('')
            elif dim == 2:
                # inset_ax = ax.inset_axes([0.0, 0.0, 1.0, 1.0])
                # for i, X_i in enumerate(X):
                #     seaborn.kdeplot(x=X_i[:, 0],
                #                     y=X_i[:, 1],
                #                     levels=10,
                #                     color=cmap(i),
                #                     fill=True,
                #                     alpha=self.scatter_params['alpha'] / 2,
                #                     ax=inset_ax)
                # inset_ax.set_xlim(ax.get_xlim())
                # inset_ax.patch.set_alpha(0)
                # inset_ax.set_facecolor(ax.get_facecolor())
                # inset_ax.set_xticks([])
                # inset_ax.set_yticks([])
                # inset_ax.set_xlabel('')
                # inset_ax.set_ylabel('')
                inset_ax_x = ax.inset_axes([0.0, 0.9, 0.9, 0.1])
                inset_ax_y = ax.inset_axes([0.9, 0.0, 0.1, 0.9])
                for i, X_i in enumerate(X):
                    seaborn.kdeplot(x=X_i[:, 0],
                                    color=cmap(i),
                                    fill=True,
                                    ax=inset_ax_x)
                    seaborn.kdeplot(y=X_i[:, 1],
                                    color=cmap(i),
                                    fill=True,
                                    ax=inset_ax_y)
                inset_ax_x.set_xlim(ax.get_xlim())
                inset_ax_x.patch.set_alpha(0)
                inset_ax_x.set_facecolor(ax.get_facecolor())
                inset_ax_x.set_xticks([])
                inset_ax_x.set_yticks([])
                inset_ax_x.set_xlabel('')
                inset_ax_x.set_ylabel('')

                inset_ax_y.set_ylim(ax.get_ylim())
                inset_ax_y.patch.set_alpha(0)
                inset_ax_y.set_facecolor(ax.get_facecolor())
                inset_ax_y.set_xticks([])
                inset_ax_y.set_yticks([])
                inset_ax_y.set_xlabel('')
                inset_ax_y.set_ylabel('')

                ax.set_xlim(xmin=None, xmax=ax.get_xlim()[0] + np.diff(ax.get_xlim()).item() / .9)
                ax.set_ylim(ymin=None, ymax=ax.get_ylim()[0] + np.diff(ax.get_ylim()).item() / .9)

                ax.annotate('KDE', (.95, .95), xycoords='axes fraction',
                            horizontalalignment='center', verticalalignment='center')

        if class_legend:
            ax.add_artist(self.class_legend(y_unique, cmap, z_unique))
        ax.grid(b=grid, **self.grid_params)
        ax.axes.get_xaxis().set_visible(grid)
        ax.axes.get_yaxis().set_visible(grid)

        if normalize:
            ax.set_xticks(np.arange(np.ceil(ax.get_xlim()[0] * 2) / 2, np.ceil(ax.get_xlim()[1] * 2) / 2, .5))
            ax.set_xticks(np.arange(np.ceil(ax.get_xlim()[0] * 2) / 2, np.ceil(ax.get_xlim()[1] * 2) / 2, .5))
            if dim == 3:
                ax.set_zticks(np.arange(np.ceil(ax.get_zlim()[0] * 2) / 2, np.ceil(ax.get_zlim()[1] * 2) / 2, .5))
        if dim == 3:
            ax.axes.get_zaxis().set_visible(grid)

        if title is not None:
            ax.set_title(title, y=self.axe3d_title_offset if dim == 3 else 1)
        return ax

    def multiview_scatter(self,
                          Xs,
                          y=None,
                          y_unique=None,
                          z=None,
                          z_unique=None,
                          dim=None,
                          normalize=True,
                          kde=False,
                          cmap=None,
                          view_titles=None,
                          title=None,
                          class_legend=True,
                          view_legend=True,
                          grid=True,
                          ax=None):
        if dim is None:
            if Xs[0].shape[1] <= 3:
                dim = Xs[0].shape[1]
            else:
                dim = 2
        n_views = len(Xs)
        n_samples = Xs[0].shape[0] if n_views > 0 else 0

        if y is None:
            y = [np.array([0 for _ in range(n_samples)]) for _ in range(n_views)]
        elif _tensor_depth(y) == 1:
            y = [y for _ in range(n_views)]
        elif _tensor_depth(y) != 2:
            raise ValueError("Labels must be 1 or 2 dimensional array.")
        with_categories = z is not None
        if with_categories:
            if _tensor_depth(z) == 1:
                z = [z for _ in range(n_views)]
            elif _tensor_depth(z) != 2:
                raise ValueError("Categories must be 1 or 2 dimensional array.")
        if y_unique is None:
            y_unique = np.unique(np.concatenate([np.unique(y_v) for y_v in y]))
        n_classes = len(y_unique)
        if z_unique is None and z is not None:
            z_unique = np.unique(np.concatenate([np.unique(z_v) for z_v in z]))

        Xs = self.embed(Xs, y, dim, normalize=normalize)
        dims = [X.shape[1] for X in Xs]
        max_dim = np.max(dims)
        if not 0 < max_dim <= 3:
            raise ValueError(f"Unable to plot {max_dim}-dimensional data.")
        if dim == 3 and z_unique is not None and len(z_unique) > 1:
            raise ValueError("Categories are not availabe for 3d plot.")

        ax = self._init_axe(ax=ax, dim=max_dim)

        if cmap is None:
            cmap = get_cmap('Spectral', n_classes)

        for v in range(n_views):
            if dims[v] < max_dim:
                Xs[v] = [np.array([np.concatenate([x, np.zeros(max_dim - dims[v])], axis=0) for x in X_i])
                         for X_i in Xs[v]]
            X_v = _group_by(Xs[v], y[v], unique_values=y_unique)
            z_v = _group_by(z[v], y[v], unique_values=y_unique) if with_categories else [None for _ in X_v]
            for i, (X_i, z_i) in enumerate(zip(X_v, z_v)):
                if z_i is not None:
                    X_i = _group_by(X_i, z_i, z_unique)
                    for j, X_ij in enumerate(X_i):
                        ax.scatter(*_orthogonal(X_ij, dim=dim),
                                   marker=_cycle(self.markers, v),
                                   color=cmap(i),
                                   s=50,
                                   **_cycle(self.borderoptions, j),
                                   **self.scatter_params)
                else:
                    ax.scatter(*_orthogonal(X_i, dim=dim),
                               marker=_cycle(self.markers, v),
                               color=cmap(i),
                               s=50,
                               **self.scatter_params)
        if kde:
            if dim == 1:
                inset_ax = ax.inset_axes([0.0, 0.8, 1.0, 0.2])
                X_all_by_cls = _group_by(np.concatenate(Xs), np.concatenate(y))
                for i, X_i in enumerate(X_all_by_cls):
                    seaborn.kdeplot(np.squeeze(X_i, 1),
                                    color=cmap(i),
                                    fill=True,
                                    ax=inset_ax)
                del X_all_by_cls
                inset_ax.set_xlim(ax.get_xlim())
                inset_ax.patch.set_alpha(0)
                inset_ax.set_facecolor(ax.get_facecolor())
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                inset_ax.set_xlabel('KDE')
                inset_ax.set_ylabel('')
            elif dim == 2:
                # inset_ax = ax.inset_axes([0.0, 0.0, 1.0, 1.0])
                # X_all_by_cls = _group_by(np.concatenate(Xs), np.concatenate(y))
                # for i, X_i in enumerate(X_all_by_cls):
                #     seaborn.kdeplot(x=X_i[:, 0],
                #                     y=X_i[:, 1],
                #                     levels=3,
                #                     color=cmap(i),
                #                     fill=True,
                #                     alpha=self.scatter_params['alpha'] / 5,
                #                     ax=inset_ax)
                # del X_all_by_cls
                # inset_ax.set_xlim(ax.get_xlim())
                # inset_ax.set_ylim(ax.get_ylim())
                # inset_ax.patch.set_alpha(0)
                # inset_ax.set_facecolor(ax.get_facecolor())
                # inset_ax.set_xticks([])
                # inset_ax.set_yticks([])
                # inset_ax.set_xlabel('')
                # inset_ax.set_ylabel('')
                inset_ax_x = ax.inset_axes([0.0, 0.9, 0.9, 0.1])
                inset_ax_y = ax.inset_axes([0.9, 0.0, 0.1, 0.9])
                X_all_by_cls = _group_by(np.concatenate(Xs), np.concatenate(y))
                for i, X_i in enumerate(X_all_by_cls):
                    seaborn.kdeplot(x=X_i[:, 0],
                                    color=cmap(i),
                                    fill=True,
                                    ax=inset_ax_x)
                    seaborn.kdeplot(y=X_i[:, 1],
                                    color=cmap(i),
                                    fill=True,
                                    ax=inset_ax_y)
                del X_all_by_cls
                inset_ax_x.set_xlim(ax.get_xlim())
                inset_ax_x.patch.set_alpha(0)
                inset_ax_x.set_facecolor(ax.get_facecolor())
                inset_ax_x.set_xticks([])
                inset_ax_x.set_yticks([])
                inset_ax_x.set_xlabel('')
                inset_ax_x.set_ylabel('')

                inset_ax_y.set_ylim(ax.get_ylim())
                inset_ax_y.patch.set_alpha(0)
                inset_ax_y.set_facecolor(ax.get_facecolor())
                inset_ax_y.set_xticks([])
                inset_ax_y.set_yticks([])
                inset_ax_y.set_xlabel('')
                inset_ax_y.set_ylabel('')

                ax.set_xlim(xmin=None, xmax=ax.get_xlim()[0] + np.diff(ax.get_xlim()).item() / .9)
                ax.set_ylim(ymin=None, ymax=ax.get_ylim()[0] + np.diff(ax.get_ylim()).item() / .9)

                ax.annotate('KDE', (.95, .95), xycoords='axes fraction',
                            horizontalalignment='center', verticalalignment='center')

        if class_legend:
            ax.add_artist(self.class_legend(y_unique, cmap, z_unique))
        if view_legend:
            ax.add_artist(self.view_legend(n_views, view_titles))
        ax.grid(b=grid, **self.grid_params)
        ax.axes.get_xaxis().set_visible(grid)
        ax.axes.get_yaxis().set_visible(grid)

        if normalize:
            ax.set_xticks(np.arange(np.ceil(ax.get_xlim()[0] * 2) / 2, np.ceil(ax.get_xlim()[1] * 2) / 2, .5))
            ax.set_xticks(np.arange(np.ceil(ax.get_xlim()[0] * 2) / 2, np.ceil(ax.get_xlim()[1] * 2) / 2, .5))
            if dim == 3:
                ax.set_zticks(np.arange(np.ceil(ax.get_zlim()[0] * 2) / 2, np.ceil(ax.get_zlim()[1] * 2) / 2, .5))
        if dim == 3:
            ax.axes.get_zaxis().set_visible(grid)

        if title is not None:
            ax.set_title(title, y=self.axe3d_title_offset if max_dim == 3 else 1)
        return ax

    def resize(self, width, height):
        """
        Resize the figure.

        Args:
            width: horizontal size in inches
            height: vertical size in inches

        Returns: None
        """
        self.fig.set_size_inches(width, height)

    def pause(self,
              title: Optional[str] = None,
              grids: Sequence[Union[str, Tuple[Union[int, slice], ...]]] = 'auto',
              size: Optional[Tuple[float, float]] = None,
              adjust: Optional[Tuple[float, ...]] = (0.06, 0.06, 0.94, 0.94, 0.15, 0.15),
              interval: float = 0.001) -> None:
        """
        Pause for rerender figure.

        Args:
            title: window title
            grids: layout of axes inside window
            size: figure size.
            adjust: tuple of (left, bottom, right, top, wspace, hspace)
            interval: time to pause in second

        Returns: None
        """
        if title is not None:
            self.fig.suptitle(title)
        self.pausing = True
        self._rearrange_fig(grids, adjust)
        if size is not None:
            self.resize(*size)
        plt.pause(interval)

    def show(self,
             title: Optional[str] = None,
             grids: Sequence[Union[str, Tuple[Union[int, slice], ...]]] = 'auto',
             figsize: Optional[Tuple[float, float]] = None,
             adjust: Optional[Tuple[float, ...]] = (0.06, 0.06, 0.94, 0.94, 0.15, 0.15),
             block: bool = True,
             clear: bool = True) -> None:
        """
        Show figure in window.

        Args:
            title: window title
            grids: layout of axes inside window
            figsize: figure size.
            adjust: tuple of (left, bottom, right, top, wspace, hspace)
            block: block the current thread. Default: True
            clear: clear the fig after showing. Default: True

        Returns: None
        """
        if title is not None:
            self.fig.suptitle(title)
        self._rearrange_fig(grids, adjust)
        if figsize is not None:
            self.resize(*figsize)
        plt.show(block=block)
        if clear:
            self._clear_fig()

    def savefig(self,
                fname: str,
                grids: Sequence[Union[str, Tuple[Union[int, slice], ...]]] = 'auto',
                figsize: Optional[Tuple[float, float]] = None,
                adjust: Optional[Tuple[float, ...]] = (0.1, 0.1, 0.95, 0.95, 0.15, 0.15),
                usetex: bool = False,
                clear: bool = False):
        """
        Save figure to file
        Args:
            fname: save file url
            grids: layout of axes inside window
            figsize: figure size
            adjust: tuple of (left, bottom, right, top, wspace, hspace)
            usetex: use latex font
            clear: clear the fig after showing. Default: True

        Returns: None
        """
        self._rearrange_fig(grids, adjust)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        if figsize is not None:
            self.resize(*figsize)
        plt.rcParams.update({'text.usetex': usetex})
        self.fig.savefig(fname)
        plt.rcParams.update({'text.usetex': False})
        if clear:
            self._clear_fig()

    def clear(self):
        """
        clear figure.

        Returns: None
        """
        self._clear_fig()

    def _init_axe(self, ax=None, dim=2):
        assert 0 < dim <= 3
        if self.fig is None:
            self.fig = plt.figure(**self.figure_params)
        if ax is None or (isinstance(ax, int) and ax >= len(self.axes)):
            new_plot_pos = (1, len(self.axes) + 1, 1)
            if dim <= 2:
                ax = self.fig.add_subplot(*new_plot_pos)
            else:
                ax = self.fig.add_subplot(*new_plot_pos, projection='3d')
                ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([self.axe3d_scale] * 3 + [1]))
                ax.patch.set_edgecolor('black')
                ax.patch.set_linewidth(1)
            self.axes.append(ax)
            plt.rc('grid', linestyle=':', color='black', alpha=0.6)
        else:
            if isinstance(ax, Axes):
                self.axes.append(ax)
            elif isinstance(ax, int):
                ax = self.axes[ax]
            else:
                raise ValueError(f'ax must be either number or Axes object, got {type(ax)}.')
            ax.clear()
        ax.set_facecolor(self.ax_params['facecolor'])
        ax.set_axisbelow(True)
        return ax

    def _rearrange_fig(self, grids='auto', adjust=(0.1, 0.1, 0.95, 0.95, 0.15, 0.15)):
        # type: (Sequence[Union[str, Tuple[Union[int, slice]]]]) -> None
        if grids in [None, 'auto']:
            gs = gridspec.GridSpec(1, len(self.axes))
            for _ in range(len(self.axes)):
                self.axes[_].set_position(gs[_].get_position(self.fig))
                self.axes[_].set_subplotspec(gs[_])
            fig_size = self.figure_params['figsize']
            self.fig.set_size_inches(fig_size[0] * len(self.axes), fig_size[1])
        else:
            for _, grid in enumerate(grids):
                grid = list(grid)
                assert len(grid) >= 3
                use_index = len(grid) == 3
                grid = grid + [None] * (6 - len(grid))
                gs = gridspec.GridSpec(*grid[:2])
                if use_index:
                    self.axes[_].set_position(gs[grid[2]].get_position(self.fig))
                    self.axes[_].set_subplotspec(gs[grid[2]])
                else:
                    self.axes[_].set_position(gs[grid[2]:grid[3], grid[4]:grid[5]].get_position(self.fig))
                    self.axes[_].set_subplotspec(gs[grid[2]:grid[3], grid[4]:grid[5]])
            # TODO: resize
        self.fig.subplots_adjust(*adjust)

    def _clear_fig(self):
        self.pausing = False
        self.fig: Optional[plt.Figure] = None
        self.axes.clear()
        plt.close(self.fig)

    def group_legend(self, groups, cmap=None, loc='upper right', fontsize=10, title="Groups", **legend_kwargs):
        if hasattr(groups, 'tolist'):
            groups = groups.tolist()
        if cmap is None:
            cmap = get_cmap('Spectral', len(groups))
        handles = [lines.Line2D([0], [0],
                                label=groups[gi],
                                marker='s',
                                markerfacecolor=cmap(gi),
                                markersize=fontsize,
                                color='black',
                                linestyle='none',
                                alpha=0.8)
                   for gi in range(len(groups))]
        legend = plt.legend(handles=handles, loc=loc, title=title, fontsize=fontsize, prop=dict(size=fontsize),
                            **{**self.legend_params, **legend_kwargs})
        plt.setp(legend.get_title(), fontsize=fontsize)
        return legend

    def class_legend(self, y_unique, cmap=None, z_unique=None,
                     loc='upper right', fontsize=10, title="Classes", **legend_kwargs):
        if hasattr(y_unique, 'tolist'):
            y_unique = y_unique.tolist()
        if cmap is None:
            cmap = get_cmap('Spectral', len(y_unique))
        handles = []
        if len(y_unique) > 1:
            handles.extend((patches.Patch(edgecolor=None, facecolor=cmap(ci), label=cls)
                            for ci, cls in enumerate(y_unique)))
        else:
            handles.append(patches.Patch(edgecolor=None, facecolor='grey', label=y_unique[0]))
        if z_unique is not None and (len(z_unique) > 1 or z_unique[0] is not None):
            handles.append(lines.Line2D([], [], linestyle='', label=''))
            handles.extend((patches.Patch(facecolor='lightgray', label=z, **_cycle(self.borderoptions, _))
                            for _, z in enumerate(z_unique)))
        legend = plt.legend(handles=handles, loc=loc, title=title, fontsize=fontsize, prop=dict(size=fontsize),
                            **{**self.legend_params, **legend_kwargs})
        plt.setp(legend.get_title(), fontsize=fontsize)
        return legend

    def category_legend(self, z_unique=None, loc='upper right', fontsize=10, title="Splits", **legend_kwargs):
        handles = [patches.Patch(facecolor='lightgray', label=z, **_cycle(self.borderoptions, _))
                   for _, z in enumerate(z_unique)]
        legend = plt.legend(handles=handles, loc=loc, title=title, fontsize=fontsize, prop=dict(size=fontsize),
                            **{**self.legend_params, **legend_kwargs})
        plt.setp(legend.get_title(), fontsize=fontsize)
        return legend

    def view_legend(self, n_views, view_titles=None, loc='lower right', fontsize=10, title="Views", **legend_kwargs):
        handles = [lines.Line2D([0], [0],
                                label='${}^{{{}}}$'.format(vi,
                                                           'st' if vi % 10 == 1
                                                           else 'nd' if vi % 10 == 2
                                                           else 'rd' if vi % 10 == 3
                                                           else 'th')
                                if view_titles is None else view_titles[vi - 1],
                                marker=_cycle(self.markers, vi - 1),
                                markerfacecolor='w',
                                markersize=fontsize,
                                color='black',
                                fillstyle='none',
                                linestyle='none',
                                alpha=0.5)
                   for vi in range(1, n_views + 1)]
        legend = plt.legend(handles=handles, loc=loc, title=title, fontsize=fontsize, prop=dict(size=fontsize),
                            **{**self.legend_params, **legend_kwargs})
        plt.setp(legend.get_title(), fontsize=fontsize)
        return legend

    @staticmethod
    def embed(Xs, ys, dim=None, normalize=False):
        ndim = _tensor_depth(Xs)
        src_dim = Xs.shape[1] if ndim == 2 else np.max([X.shape[1] for X in Xs]) if len(Xs) > 0 else 0
        if 0 < src_dim <= 3 and (dim is None or dim == src_dim):
            if _tensor_depth(Xs) == 2:
                return StandardScaler().fit_transform(Xs) if normalize else Xs
            else:
                n_views = len(Xs)
                X_all = np.concatenate([X for X in Xs])
                X_embed = StandardScaler().fit_transform(X_all) if normalize else X_all
                return [X_embed[int(len(X_embed) / n_views * _):int(len(X_embed) / n_views * (_ + 1))]
                        for _ in range(n_views)]
        elif dim is None:
            dim = 2

        if _tensor_depth(Xs) == 2:
            X_embed = TSNE(n_components=dim, perplexity=Xs.shape[0] / 2).fit_transform(Xs, ys)
            return StandardScaler().fit_transform(X_embed)
        else:
            n_views = len(Xs)
            if _tensor_depth(ys) == 1:
                ys = [ys for _ in range(n_views)]
            n_views = len(Xs)
            X_all = np.concatenate([X for X in Xs])
            y_all = np.concatenate([y for y in ys])
            X_embed = TSNE(n_components=dim, perplexity=X_all.shape[0] / 2).fit_transform(X_all, y_all)
            X_embed = StandardScaler().fit_transform(X_embed)
            return [X_embed[int(len(X_embed) / n_views * _):int(len(X_embed) / n_views * (_ + 1))]
                    for _ in range(n_views)]
