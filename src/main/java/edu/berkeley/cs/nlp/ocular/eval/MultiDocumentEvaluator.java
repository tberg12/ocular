package edu.berkeley.cs.nlp.ocular.eval;

import java.util.Map;

import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModel;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public interface MultiDocumentEvaluator {

	public void printTranscriptionWithEvaluation(int iter, int batchId,
			CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm, Map<String, CharacterTemplate> font);
	
	/**
	 * No-op version of an evaluator 
	 */
	public static class NoOpMultiDocumentEvaluator implements MultiDocumentEvaluator {
		public void printTranscriptionWithEvaluation(int iter, int batchId,
				CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm, Map<String, CharacterTemplate> font) {}
	}
}
