import os
import qiime2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _convergence_plot(model, baseline, ax0, ax1):
    iterations = np.array(model['iteration'])

    ax0.plot(iterations[1:],
             np.array(model['cross-validation'].values)[1:],
             label='model')
    ax0.set_ylabel('Cross validation score', fontsize=14)
    ax0.set_xlabel('# Iterations', fontsize=14)

    ax1.plot(iterations[1:],
             np.array(model['loss'])[1:], label='model')
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_xlabel('# Iterations', fontsize=14)

    if baseline is not None:
        iterations = baseline['iteration']

        ax0.plot(iterations[1:],
                 np.array(baseline['cross-validation'].values)[1:],
                 label='baseline')
        ax0.set_ylabel('Cross validation score', fontsize=14)
        ax0.set_xlabel('# Iterations', fontsize=14)
        ax0.legend()

        ax1.plot(iterations[1:],
                 np.array(baseline['loss'])[1:], label='baseline')
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.set_xlabel('# Iterations', fontsize=14)
        ax1.legend()


def _summarize(output_dir: str, model: pd.DataFrame,
               baseline: pd.DataFrame = None):

    """ Helper method for generating summary pages
    Parameters
    ----------
    output_dir : str
       Name of output directory
    model : pd.DataFrame
       Model summary with column names
       ['loss', 'cross-validation']
    baseline : pd.DataFrame
       Baseline model summary with column names
       ['loss', 'cross-validation']. Defaults to None (i.e. if only a single
       set of model stats will be summarized).
    Note
    ----
    There may be synchronizing issues if different summary intervals
    were used between analyses. For predictable results, try to use the
    same summary interval.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    if baseline is None:
        _convergence_plot(model, None, ax[0], ax[1])
        q2 = None
    else:

        _convergence_plot(model, baseline, ax[0], ax[1])

        # this provides a pseudo-r2 commonly provided in the context
        # of logistic / multinomail model (proposed by Cox & Snell)
        # http://www3.stat.sinica.edu.tw/statistica/oldpdf/a16n39.pdf
        end = min(10, len(model.index))
        # trim only the last 10 numbers

        # compute a q2 score, which is commonly used in
        # partial least squares for cross validation
        l0 = np.mean(baseline['cross-validation'][-end:])
        lm = np.mean(model['cross-validation'][-end:])
        q2 = 1 - lm / l0

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'convergence-plot.svg'))
    fig.savefig(os.path.join(output_dir, 'convergence-plot.pdf'))

    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        index_f.write('<h1>Convergence summary</h1>\n')
        index_f.write(
            "<p>If you don't see anything in these plots, you probably need "
            "to decrease your <kbd>--p-summary-interval</kbd>. Try setting "
            "<kbd>--p-summary-interval 1</kbd>, which will record the loss at "
            "every second.</p>\n"
        )

        if q2 is not None:
            index_f.write(
                '<p><strong>'
                '<a href="https://github.com/biocore/mmvec#explaining-q2">'
                'Pseudo Q-squared:</a></strong> %f</p>\n' % q2
            )

        index_f.write(
            '<img src="convergence-plot.svg" alt="convergence_plots">'
        )
        index_f.write('<a href="convergence-plot.pdf">')
        index_f.write('Download as PDF</a><br>\n')


def summarize_single(output_dir: str, model_stats: qiime2.Metadata):
    _summarize(output_dir, model_stats.to_dataframe())


def summarize_paired(output_dir: str,
                     model_stats: qiime2.Metadata,
                     baseline_stats: qiime2.Metadata):
    _summarize(output_dir,
               model_stats.to_dataframe(),
               baseline_stats.to_dataframe())
