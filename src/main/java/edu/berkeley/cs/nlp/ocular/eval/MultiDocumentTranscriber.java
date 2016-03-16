package edu.berkeley.cs.nlp.ocular.eval;

import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface MultiDocumentTranscriber {

	public void transcribe(Font font, CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm);
	public void transcribe(int iter, int batchId, Font font, CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm);
	
	/**
	 * No-op evaluator implementation
	 */
	public static class NoOpMultiDocumentTranscriber implements MultiDocumentTranscriber {
		public void transcribe(Font font, CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm) {}
		public void transcribe(int iter, int batchId, Font font, CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm) {}
	}
}
