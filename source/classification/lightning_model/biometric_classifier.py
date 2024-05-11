from classification.lightning_model import Classifier


class BiometricClassifier(Classifier):
    """Classifier for performing biometry experiment."""

    def on_test_epoch_end(self) -> None:
        """Finish test epoch and log all metrics results and loss."""
        results = self._epoch_end(self.epoch_results)
        results = self._eval_epoch_end(results)
        results = self._vote(results)["measurements"]
        if self.hparams.prefix is not None:
            results = self._add_prefix_to_metrics(f"{self.hparams.prefix}/", results)
        logs = self._add_prefix_to_metrics("test/", results)

        self.log_dict(logs)
        self.epoch_results = []
        self.trainer.logger.finalize("success")
