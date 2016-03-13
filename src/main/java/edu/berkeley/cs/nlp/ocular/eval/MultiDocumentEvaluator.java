package edu.berkeley.cs.nlp.ocular.eval;

import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModel;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface MultiDocumentEvaluator {

	public void printTranscriptionWithEvaluation(int iter, int batchId,
			CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm, Font font);
	
	/**
	 * No-op evaluator implementation
	 */
	public static class NoOpMultiDocumentEvaluator implements MultiDocumentEvaluator {
		public void printTranscriptionWithEvaluation(int iter, int batchId,
				CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm, Font font) {}
	}
}
