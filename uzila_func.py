import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from warnings import warn
    
class pre_process:

    def PlotHighCorr(data, target, n_largest = 10):
        corrmat = data.corr().abs()     # pearson correlation, is it suitable?
        corrmat_largest = corrmat.nlargest(n_largest, target)
        corrmat_largest_cols = corrmat_largest[target].index
        corrmat_largest = corrmat_largest[corrmat_largest_cols]
        plt.figure(figsize=[10,10])
        sns.heatmap(corrmat_largest, vmin=0, vmax=1, annot=True, square=True, fmt='.2f')
        plt.show()
        return corrmat_largest_cols

    def PlotOutlier(data, data_outlier, cols, target):
        plt.figure(figsize=(15, 30))
        for i, c in enumerate(cols):
            plt.subplot(np.ceil(len(cols)/2), 2, i+1)
            plt.scatter(c, target, data=data)
            plt.scatter(c, target, data=data_outlier, c='r')
            plt.xlabel(c)
            plt.ylabel(target)
        plt.show()

    def MissVal(df):
        miss_sum = df.isnull().sum()
        miss_pct = df.isnull().sum() / df.isnull().count() * 100
        miss_val = pd.concat([miss_sum, miss_pct], axis=1, keys=['Total', 'Percentage'])
        miss_val = miss_val[miss_val.Total>0].sort_values(by='Total', ascending=False)
        return(miss_val)

    def TableMissVal(train, test):
        miss_train = pre_process.MissVal(train)
        miss_test = pre_process.MissVal(test)
        miss_all = pd.concat([miss_train, miss_test], axis=1, sort=False)
        miss_all.columns = pd.MultiIndex.from_product([['miss_train', 'miss_test'], ['Total', 'Percentage']])
        miss_all.sort_values(by=[('miss_train', 'Total'), ('miss_test', 'Total')], ascending=False, inplace=True)
        print(miss_all)
        return(miss_all)

class post_process:
    
    def PlotLearningCurve(estimator, title, X, y, score, ylim=None, cv=None,
                          n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=score, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(-train_scores, axis=1)
        train_scores_std = np.std(-train_scores, axis=1)
        test_scores_mean = np.mean(-test_scores, axis=1)
        test_scores_std = np.std(-test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        print('Cross-validation score:', test_scores_mean[-1])
        return plt

class multilabel:
    
    def multilabel_sample(y, size=1000, min_count=5, seed=None):
        """ Takes a matrix of binary labels `y` and returns
            the indices for a sample of size `size` if
            `size` > 1 or `size` * len(y) if size =< 1.

            The sample is guaranteed to have > `min_count` of
            each label.
        """
        try:
            if (np.unique(y).astype(int) != np.array([0, 1])).all():
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError('multilabel_sample only works with binary indicator matrices')

        if (y.sum(axis=0) < min_count).any():
            raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')

        if size <= 1:
            size = np.floor(y.shape[0] * size)

        if y.shape[1] * min_count > size:
            msg = "Size less than number of columns * min_count, returning {} items instead of {}."
            warn(msg.format(y.shape[1] * min_count, size))
            size = y.shape[1] * min_count

        rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))

        if isinstance(y, pd.DataFrame):
            choices = y.index
            y = y.values
        else:
            choices = np.arange(y.shape[0])

        sample_idxs = np.array([], dtype=choices.dtype)

        # first, guarantee > min_count of each label
        for j in range(y.shape[1]):
            label_choices = choices[y[:, j] == 1]
            label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
            sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])

        sample_idxs = np.unique(sample_idxs)

        # now that we have at least min_count of each, we can just random sample
        sample_count = int(size - sample_idxs.shape[0])

        # get sample_count indices from remaining choices
        remaining_choices = np.setdiff1d(choices, sample_idxs)
        remaining_sampled = rng.choice(remaining_choices,
                                       size=sample_count,
                                       replace=False)

        return np.concatenate([sample_idxs, remaining_sampled])

    def multilabel_sample_dataframe(df, labels, size, min_count=5, seed=None):
        """ Takes a dataframe `df` and returns a sample of size `size` where all
            classes in the binary matrix `labels` are represented at
            least `min_count` times.
        """
        idxs = multilabel.multilabel_sample(labels, size=size, min_count=min_count, seed=seed)
        return df.loc[idxs]

    def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):
        """ Takes a features matrix `X` and a label matrix `Y` and
            returns (X_train, X_test, Y_train, Y_test) where all
            classes in Y are represented at least `min_count` times.
        """
        index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])

        test_set_idxs = multilabel.multilabel_sample(Y, size=size, min_count=min_count, seed=seed)
        train_set_idxs = np.setdiff1d(index, test_set_idxs)

        test_set_mask = index.isin(test_set_idxs)
        train_set_mask = ~test_set_mask

        return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])
