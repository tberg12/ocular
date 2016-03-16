package edu.berkeley.cs.nlp.ocular.gsm;

import java.io.Serializable;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface GlyphSubstitutionModel extends Serializable {

	public double glyphProb(int language, int lmChar, GlyphChar glyphChar);

}
