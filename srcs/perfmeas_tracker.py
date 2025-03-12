#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/01/30
#  Modified Date: 2023/06/23
#
# ----------------------------------------------------------------------------

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
)

##############################################################################
privacy_classes = {
    "binary": {0: "private", 1: "public"},
    "binary-privacyalert": {0: "public", 1: "private"},
}


##############################################################################
class AverageMeter(object):
    """ """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.tot = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.tot += val * n
        self.count += n
        self.avg = self.tot / self.count

    def get_val(self):
        return self.val

    def get_tot(self):
        return self.tot

    def get_count(self):
        return self.count

    def get_average(self):
        return self.avg

    def get_measure(self, str_measure="avg"):
        if str_measure == "val":
            measure = self.get_val()
        if str_measure == "tot":
            measure = self.get_tot()
        if str_measure == "count":
            measure = self.get_count()
        if str_measure == "avg":
            measure = self.get_average()

        return measure

    def print_measure(self, str_measure="avg"):
        measure = self.get_measure(str_measure)
        print(str_measure + ": " + str(measure))


##############################################################################
class PerformanceMeasuresTracker(object):
    """Class to compute and track performance measure for binary classification.

    Performance measures are computed for the privacy and biology use cases.
    Current implementation consider the GIPS and BRCA datasets chosen for the
    common architecture.
    """

    def __init__(self, dataset="IPD", n_cls=2, beta=2):
        """ """
        assert dataset in ["GIPS", "IPD", "PicAlert", "PrivacyAlert", "VISPR"]
        self.dataset = dataset

        self.n_classes = n_cls
        self.lables = self.get_lables_list()

        self.get_measures_names()

        self.beta = beta

        # True positives, false positives, true negatives, false negatives for
        # binary classification
        self.tp = 0  # True positives
        self.tn = 0  # True negatives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives

        # Overall scores
        self.beta_f1 = 0  # Beta F1-score
        self.acc_b = 0  # Balanced accuracy

        self.cm = None  # confusion matrix
        self.cr = None  # classification report

    def compute_confusion_matrix(self, gts, preds):
        """ """
        self.cm = confusion_matrix(gts, preds, normalize="true")

    def compute_all_metrics(self, gts, preds):
        """ """
        self.compute_confusion_matrix(gts, preds)

        self.cr = classification_report(
            gts,
            preds,
            output_dict=True,
            target_names=self.lables,
            zero_division=0,
        )

        # Overall scores
        self.acc_b = balanced_accuracy_score(gts, preds)

        avg_type_str = "weighted"

        if self.n_classes == 2:
            self.tn, self.fp, self.fn, self.tp = self.cm.ravel()
            avg_type_str = "binary"

        self.beta_f1 = fbeta_score(
            gts, preds, beta=self.beta, average=avg_type_str, zero_division=0
        )

    def is_classification_report(self):
        if self.cr is not None:
            return True
        else:
            return False

    def get_lables_list(self):
        labels = []

        if self.n_classes == 2:
            dataset_cls = privacy_classes["binary"]

        if self.n_classes == 3:
            dataset_cls = privacy_classes["ternary"]

        if self.n_classes == 5:
            dataset_cls = privacy_classes["quinary"]

        for idx, label in dataset_cls.items():
            labels.append(label)

        return labels

    def get_measures_names(self):
        self.measures_names = [
            "precision",
            "recall",
            "macro_f1_score",
            "weighted_f1_score",
            "accuracy",
            "balanced_accuracy",
            "confusion_matrix",
            "beta_f1_score",
        ]

        self.measures_names.append("precision_0")
        self.measures_names.append("recall_0")
        self.measures_names.append("f1_score_0")
        self.measures_names.append("precision_1")
        self.measures_names.append("recall_1")
        self.measures_names.append("f1_score_1")

        if self.n_classes == 3 or self.n_classes == 5:
            self.measures_names.append("precision_2")
            self.measures_names.append("recall_2")
            self.measures_names.append("f1_score_2")

        if self.n_classes == 5:
            self.measures_names.append("precision_3")
            self.measures_names.append("recall_3")
            self.measures_names.append("f1_score_3")
            self.measures_names.append("precision_4")
            self.measures_names.append("recall_4")
            self.measures_names.append("f1_score_4")

    def get_classification_report(self):
        return self.cr

    def get_confusion_matrix(self):
        return self.cm

    def get_measure_cls(self, category, measure):
        return self.cr[category][measure]

    def get_macro_f1_score(self):
        return self.cr["macro avg"]["f1-score"]

    def get_beta_f1_score(self):
        return self.beta_f1

    def get_accuracy(self):
        return self.cr["accuracy"]

    def get_balanced_accuracy(self):
        return self.acc_b

    def get_weighted_f1score(self):
        return self.cr["weighted avg"]["f1-score"]

    def get_precision_overall(self):
        return self.cr["macro avg"]["precision"]

    def get_recall_overall(self):
        return self.cr["macro avg"]["recall"]

    def get_measure(self, measure_name):
        assert measure_name in self.measures_names

        if measure_name == "precision_0":
            return self.get_measure_cls(self.lables[0], "precision")
        if measure_name == "recall_0":
            return self.get_measure_cls(self.lables[0], "recall")
        if measure_name == "f1_score_0":
            return self.get_measure_cls(self.lables[0], "f1-score")
        if measure_name == "precision_1":
            return self.get_measure_cls(self.lables[1], "precision")
        if measure_name == "recall_1":
            return self.get_measure_cls(self.lables[1], "recall")
        if measure_name == "f1_score_1":
            return self.get_measure_cls(self.lables[1], "f1-score")

        if measure_name == "precision_2":
            return self.get_measure_cls(self.lables[2], "precision")
        if measure_name == "recall_2":
            return self.get_measure_cls(self.lables[2], "recall")
        if measure_name == "f1_score_2":
            return self.get_measure_cls(self.lables[2], "f1-score")

        if measure_name == "precision_3":
            return self.get_measure_cls(self.lables[3], "precision")
        if measure_name == "recall_3":
            return self.get_measure_cls(self.lables[3], "recall")
        if measure_name == "f1_score_3":
            return self.get_measure_cls(self.lables[3], "f1-score")

        if measure_name == "precision_4":
            return self.get_measure_cls(self.lables[4], "precision")
        if measure_name == "recall_4":
            return self.get_measure_cls(self.lables[4], "recall")
        if measure_name == "f1_score_4":
            return self.get_measure_cls(self.lables[4], "f1-score")

        if measure_name == "macro_f1_score":
            return self.get_macro_f1_score()
        if measure_name == "beta_f1_score":
            return self.get_beta_f1_score()
        if measure_name == "weighted_f1_score":
            return self.get_weighted_f1score()
        if measure_name == "accuracy":
            return self.get_accuracy()
        if measure_name == "balanced_accuracy":
            return self.get_balanced_accuracy()
        if measure_name == "precision":
            return self.get_precision_overall()
        if measure_name == "recall":
            return self.get_recall_overall()

    def set_measure_cls(self, category, measure, value):
        self.cr[category][measure] = value

    def set_macro_f1_score(self, value):
        self.cr["macro avg"]["f1-score"] = value

    def set_beta_f1_score(self, value):
        self.beta_f1 = value

    def set_accuracy(self, value):
        self.cr["accuracy"] = value

    def set_balanced_accuracy(self, value):
        self.acc_b = value

    def set_weighted_f1score(self, value):
        self.cr["weighted avg"]["f1-score"] = value

    def set_beta(self, beta):
        self.beta = beta

    def set_precision_overall(self, value):
        self.cr["macro avg"]["precision"] = value

    def set_recall_overall(self, value):
        self.cr["macro avg"]["recall"] = value

    def set_classification_report(self, cr_new):
        self.cr = cr_new

    def set_measure(self, measure_name, measure_val):
        assert measure_name in self.measures_names

        if measure_name == "precision_0":
            self.set_measure_cls(self.lables[0], "precision", measure_val)
        if measure_name == "recall_0":
            self.set_measure_cls(self.lables[0], "recall", measure_val)
        if measure_name == "f1_score_0":
            self.set_measure_cls(self.lables[0], "f1-score", measure_val)
        if measure_name == "precision_1":
            self.set_measure_cls(self.lables[1], "precision", measure_val)
        if measure_name == "recall_1":
            self.set_measure_cls(self.lables[1], "recall", measure_val)
        if measure_name == "f1_score_1":
            self.set_measure_cls(self.lables[1], "f1-score", measure_val)

        if measure_name == "precision_2":
            self.set_measure_cls(self.lables[2], "precision", measure_val)
        if measure_name == "recall_2":
            self.set_measure_cls(self.lables[2], "recall", measure_val)
        if measure_name == "f1_score_2":
            self.set_measure_cls(self.lables[2], "f1-score", measure_val)

        if measure_name == "precision_3":
            self.set_measure_cls(self.lables[3], "precision", measure_val)
        if measure_name == "recall_3":
            self.set_measure_cls(self.lables[3], "recall", measure_val)
        if measure_name == "f1_score_3":
            self.set_measure_cls(self.lables[3], "f1-score", measure_val)

        if measure_name == "precision_4":
            self.set_measure_cls(self.lables[4], "precision", measure_val)
        if measure_name == "recall_4":
            self.set_measure_cls(self.lables[4], "recall", measure_val)
        if measure_name == "f1_score_4":
            self.set_measure_cls(self.lables[4], "f1-score", measure_val)

        # Overall
        if measure_name == "macro_f1_score":
            self.set_macro_f1_score(measure_val)
        if measure_name == "beta_f1_score":
            self.set_beta_f1_score(measure_val)
        if measure_name == "weighted_f1_score":
            self.set_weighted_f1score(measure_val)
        if measure_name == "accuracy":
            self.set_accuracy(measure_val)
        if measure_name == "balanced_accuracy":
            self.set_balanced_accuracy(measure_val)
        if measure_name == "precision":
            self.set_precision_overall(measure_val)
        if measure_name == "recall":
            self.set_recall_overall(measure_val)

    def get_all_metrics(self, mode="", percentage=False):
        if percentage:
            self.convert_to_percentages()

        if mode == "dict":
            metrics_dict = dict()

            for measure in self.measures_names:
                if percentage:
                    metrics_dict[measure] = self.get_measure(measure) * 100
                else:
                    metrics_dict[measure] = self.get_measure(measure)

            return metrics_dict
        else:
            if percentage:
                return (
                    self.get_measure("accuracy") * 100,
                    self.get_measure("precision_0") * 100,
                    self.get_measure("recall_0") * 100,
                    self.get_measure("precision_1") * 100,
                    self.get_measure("recall_1") * 100,
                    self.get_confusion_matrix(),
                    self.get_measure("macro_f1_score") * 100,
                )
            else:
                return (
                    self.get_measure("accuracy"),
                    self.get_measure("precision_0"),
                    self.get_measure("recall_0"),
                    self.get_measure("precision_1"),
                    self.get_measure("recall_1"),
                    self.get_confusion_matrix(),
                    self.get_measure("macro_f1_score"),
                )

    def print_metrics(self):
        """ """
        r0 = self.get_measure("recall_0")
        r1 = self.get_measure("recall_1")
        prec = self.get_precision_overall()
        # rec = self.get_recall_overall()
        acc = self.get_accuracy()
        acc_b = self.get_balanced_accuracy()
        w_f1 = self.get_weighted_f1score()
        macro_f1 = self.get_macro_f1_score()

        # print(
        #     "P (%): {:.2f} | BA (%): {:.2f} | UBA (%): {:.2f} | wF1 (%): {:.2f} | MF1 (%): {:.2f}\n".format(
        #         prec * 100,
        #         acc_b * 100,
        #         acc * 100,
        #         w_f1 * 100,
        #         macro_f1 * 100,
        #     )
        # )

        print(
            "R0 (%): {:.2f} | R1 (%): {:.2f} | P (%): {:.2f} | BA (%): {:.2f} | ACC (%): {:.2f}\n".format(
                r0 * 100,
                r1 * 100,
                prec * 100,
                acc_b * 100,
                acc * 100,
            )
        )

    def write_metrics_to_log(self, log):
        """ """

        prec = self.get_precision_overall()
        # rec = self.get_recall_overall()
        acc = self.get_accuracy()
        acc_b = self.get_balanced_accuracy()
        w_f1 = self.get_weighted_f1score()
        macro_f1 = self.get_macro_f1_score()

        log.write("P (%): {:.2f} | ".format(prec * 100))
        # log.write("R (%): {:.2f} | ".format(rec * 100))
        log.write("BA (%): {:.2f} | ".format(acc_b * 100))
        log.write("UBA (%): {:.2f} | ".format(acc * 100))
        log.write("wF1 (%): {:.2f}\n".format(w_f1 * 100))
        log.write("MF1 (%): {:.2f}\n".format(macro_f1 * 100))

        log.write(str(self.cm) + "\n")
