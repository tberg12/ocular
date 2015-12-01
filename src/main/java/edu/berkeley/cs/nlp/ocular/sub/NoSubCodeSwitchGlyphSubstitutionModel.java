package edu.berkeley.cs.nlp.ocular.sub;

import java.io.Serializable;

import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;
import indexer.Indexer;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class NoSubCodeSwitchGlyphSubstitutionModel implements CodeSwitchGlyphSubstitutionModel, Serializable {
	private static final long serialVersionUID = 1L;

	private CodeSwitchLanguageModel lm;
	private SingleGlyphSubstitutionModel sgsm;
	
	public NoSubCodeSwitchGlyphSubstitutionModel(CodeSwitchLanguageModel lm) {
		this.lm = lm;
		this.sgsm = new NoSubSingleGlyphSubstitutionModel();
	}

	public Indexer<String> getLanguageIndexer() {
		return lm.getLanguageIndexer();
	}
	
	public SingleGlyphSubstitutionModel get(int language) {
		return sgsm;
	}
	
	public double logLanguagePrior(int language) {
		return Math.log(lm.languagePrior(language));
	}
	
	private static class NoSubSingleGlyphSubstitutionModel implements SingleGlyphSubstitutionModel, Serializable {
		private static final long serialVersionUID = 1L;
		
		public double logGlyphProb(GlyphType prevGlyphChar, int prevLmChar, int lmChar, GlyphChar glyphChar) {
			return ((lmChar == glyphChar.templateCharIndex) ? 0.0 : Double.NEGATIVE_INFINITY);
		}
		
	}

}
