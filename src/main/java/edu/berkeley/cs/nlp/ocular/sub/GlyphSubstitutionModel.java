package edu.berkeley.cs.nlp.ocular.sub;

import java.io.Serializable;

import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public interface GlyphSubstitutionModel extends Serializable {

	public double glyphProb(int language, GlyphType prevGlyphType, int prevLmChar, int lmChar, GlyphChar glyphChar);

}
