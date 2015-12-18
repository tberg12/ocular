package edu.berkeley.cs.nlp.ocular.sub;

import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class NoSubGlyphSubstitutionModel implements GlyphSubstitutionModel {
	private static final long serialVersionUID = 1L;

	public NoSubGlyphSubstitutionModel() {
	}
	
	public double glyphProb(int language, GlyphType prevGlyphType, int prevLmChar, int lmChar, GlyphChar glyphChar) {
		return (glyphChar.glyphType == GlyphType.NORMAL_CHAR && lmChar == glyphChar.templateCharIndex) ? 1.0 : 0.0;
	}
	
}
