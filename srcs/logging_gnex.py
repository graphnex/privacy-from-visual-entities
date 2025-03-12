#!/usr/bin/env python
#
# Utilities for customised logging
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/06/28
# Modified Date: 2023/08/07
#
# ----------------------------------------------------------------------------

from datetime import datetime

#############################################################################
LOG_LINE_LENGHT = 58  # CONSTANT FOR THE NUMBER OF '-' TO WRITE IN THE LOG FILE


class Logging(object):
    def __init__(self):
        self.log = None

    def initialise(self, filename):
        """Create testing log and initialise the preamble."""
        # filename = self.get_filename(extension=".txt", suffix="_testing_log")

        # self.log = open(os.path.join(self.checkpoint_dir, filename), "w")

        self.log = open(filename, "w")

        self.log.write("".join(["-" * LOG_LINE_LENGHT]) + "\n")

        self.log.write("Experiment initiated on:")
        self.log.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\n\n")

    def write_preamble(self, model_name, n_out_classes):
        """ """
        self.log.write("Model name: " + model_name + "\n")

        self.log.write(
            "Number of output classes: {:d}\n".format(n_out_classes)
        )

        self.log.write("".join(["-" * LOG_LINE_LENGHT]) + "\n")

    def write_training_parameters(
        self, batch_sz_train, batch_sz_val, num_epochs, fold_id, partition
    ):
        """ """
        # self.log.write("Learning rate: " + str(self.scheduler) + "\n")
        self.log.write("Batch_size (training): " + str(batch_sz_train) + "\n")
        self.log.write("Batch_size (validation): " + str(batch_sz_val) + "\n")
        self.log.write("Number of epochs: " + str(num_epochs) + "\n")

        if partition == "crossval":
            self.log.write("Fold {:d}\n".format(fold_id))

    def write_line(self):
        self.log.write("".join(["-" * LOG_LINE_LENGHT]) + "\n")

    def write_info(self, info_string):
        """ """
        self.log.write(info_string)

    def write_epoch_info(self, mode="Training"):
        # Save information about the validation (i.e., headers of unbalanced accuracy,
        # balanced, accuracy, and weighted F1-score, all in percentages)
        self.write_line()
        self.log.write("\n" + mode + "\n")
        self.log.write(
            "P (%) | R (%) | UBA (%) | BA (%) | wF1 (%) | MF1 (%)\n"
        )

    def write_batch_info(self, perf_tracker):
        """ """
        pre = perf_tracker.get_precision_overall() * 100
        rec = perf_tracker.get_recall_overall() * 100
        b_acc = perf_tracker.get_balanced_accuracy() * 100
        acc = perf_tracker.get_accuracy() * 100
        w_f1 = perf_tracker.get_weighted_f1score() * 100
        macro_f1 = perf_tracker.get_macro_f1_score() * 100

        self.log.write(
            "{:6.2f} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f}\n".format(
                pre, rec, acc, b_acc, w_f1, macro_f1
            )
        )

    def get_log(self):
        return self.log

    def write_ending(self):
        """ """
        self.log.write("Experiment terminated on: ")
        self.log.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

        self.log.close()
