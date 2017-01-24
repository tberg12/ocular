package edu.berkeley.cs.nlp.ocular.eval;

import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat;
import edu.berkeley.cs.nlp.ocular.model.DecodeState;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface SingleDocumentEvaluatorAndOutputPrinter {
	
	public Tuple2<Map<String, EvalSuffStats>,Map<String, EvalSuffStats>> evaluateAndPrintTranscription(int iter, int batchId,
			Document doc,
			DecodeState[][] decodeStates,
			String inputDocPath, String outputPath, Set<OutputFormat> outputFormats,
			CodeSwitchLanguageModel lm);

}
