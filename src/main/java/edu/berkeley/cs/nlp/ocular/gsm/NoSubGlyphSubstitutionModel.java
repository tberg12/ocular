package edu.berkeley.cs.nlp.ocular.gsm;

import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class NoSubGlyphSubstitutionModel implements GlyphSubstitutionModel {
	private static final long serialVersionUID = 1L;

	public NoSubGlyphSubstitutionModel() {
	}
	
	public double glyphProb(int language, int lmChar, GlyphChar glyphChar) {
		return (glyphChar.glyphType == GlyphType.NORMAL_CHAR && lmChar == glyphChar.templateCharIndex) ? 1.0 : 0.0;
	}
	
}
