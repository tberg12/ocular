package edu.berkeley.cs.nlp.ocular.sub;

import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public interface SingleGlyphSubstitutionModel {

	public double logGlyphProb(GlyphType prevGlyphChar, int prevLmChar, int lmChar, GlyphChar glyphChar);
	
}
