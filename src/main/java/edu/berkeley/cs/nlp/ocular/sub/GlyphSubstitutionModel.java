package edu.berkeley.cs.nlp.ocular.sub;

import java.io.Serializable;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public interface GlyphSubstitutionModel extends Serializable {

	public double glyphProb(int language, int lmChar, GlyphChar glyphChar);

}
