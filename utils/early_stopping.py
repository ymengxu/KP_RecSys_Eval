class EarlyStopping:
    def __init__(self, consecutive_eval_threshold=5):
        """stop the training process if for {consecutive_eval_threshold} consecutive evaluation steps, 
        there is no improvement in the evaluation metric
        
        Args:
            consecutive_eval_threshold (int): how many consecutive validation steps with no improvement before stopping the training process
                                              default 5    
        """
        self.consecutive_eval_threshold = consecutive_eval_threshold
        self.best_evaluation_score = 0
        self.best_epoch = 0
        self.current_evaluation_score = 0
        self.consecutive_evaluation_index = 0  # how many evaluation steps so far with no improvement

    def log(self, epoch_id:int, evaluation_score):
        """log the current evaluation score, returns True if decide to stop training"""
        stop_flag = False
        
        self.current_evaluation_score = evaluation_score
        if self.current_evaluation_score > self.best_evaluation_score:
            self.best_evaluation_score = self.current_evaluation_score
            self.best_epoch = epoch_id
            # restart counting the consecutive evaluation steps
            self.consecutive_evaluation_index = 0
        else:
            self.consecutive_evaluation_index += 1
        
        if self.consecutive_evaluation_index >= self.consecutive_eval_threshold:
            stop_flag = True
        
        return stop_flag
        