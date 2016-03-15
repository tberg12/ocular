package edu.berkeley.cs.nlp.ocular.eval;

import java.util.Map;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface SingleDocumentEvaluatorAndOutputPrinter {
	
	public Tuple2<Map<String, EvalSuffStats>,Map<String, EvalSuffStats>> evaluateAndPrintTranscription(int iter, int batchId,
			Document doc,
			TransitionState[][] decodeStates, int[][] decodeWidths,
			String inputDocPath, String outputPath);

}
