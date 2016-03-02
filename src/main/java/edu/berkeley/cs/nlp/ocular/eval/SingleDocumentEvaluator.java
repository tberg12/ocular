package edu.berkeley.cs.nlp.ocular.eval;

import java.util.List;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface SingleDocumentEvaluator {
	
	public void printTranscriptionWithEvaluation(int iter, int batchId,
			Document doc,
			TransitionState[][] decodeStates, int[][] decodeWidths,
			String inputDocPath, String outputPath,
			List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals,
			List<Tuple2<String, Map<String, EvalSuffStats>>> allLmEvals);

	/**
	 * No-op version of an evaluator 
	 */
	public static class NoOpDocumentEvaluator implements SingleDocumentEvaluator {
		public void printTranscriptionWithEvaluation(int iter, int batchId,
				Document doc,
				TransitionState[][] decodeStates, int[][] decodeWidths,
				String inputDocPath, String outputPath,
				List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals,
				List<Tuple2<String, Map<String, EvalSuffStats>>> allLmEvals) {}
	}
}
