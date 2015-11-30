package edu.berkeley.cs.nlp.ocular.model;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.model.GlyphChar.GlyphType;

/**
 * Probability of the model choosing to generate the `glyphCharToRender` 
 * as a glyph given that the language model wants to generate the `lmChar`,
 * and whether the previous `glyphCharToRender` was an elision-tilde 
 * character or was elided itself.
 * For example:
 * 
 *       Language:    s -> s -> s -> s -> s
 *                    |    |    |    |    |
 *       LM chars:    u -> n -> a -> m -> e
 *                    | \  | \  | \  | \  |
 *    Glyph chars:    v -> n -> ã -> # -> e
 * 
 * would mean that the LM wants to generate the string of characters "uname" 
 * because it is likely given the (modern spellings in the) LM training data,
 * but the printer chose to render this string as "vnãe", changing the 'u' to
 * a 'v', the 'a' to a 'ã', and eliding the 'm' (here '#' is standing in for
 * a zero-width rendering).  Thus, we would need to compute, for example: 
 *   
 *         P( v | u, ... )
 *         P( n | n, v )
 *         P( ã | a, n )
 *         P( m | m, ã )
 *         P( e | e, # )
 *         
 * The "previous glyph" context is useful because we, for example, will only
 * allow zero-width characters (elisions) after elision-tilde letters (those
 * letters for which the tilde was added (as a substituted letter) just for 
 * the elision).
 * 
 * 
 * 
 * P( glyph[c,ti,el] | language, lmChar, prevLmChar, prevGlyph[pc,pti,pel] ) =
 *                  P( language )
 *     P( prevGlyph[pc,pti,pel] | language )
 *                P( prevLmChar | language, prevGlyph[pc,pti,pel] )
 *                    P( lmChar | language, prevGlyph[pc,pti,pel], prevLmChar )
 *            P( glyph[c,ti,el] | language, prevGlyph[pc,pti,pel], prevLmChar, lmChar )
 *
 * 
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class GlyphProbModel implements Serializable {
	private static final long serialVersionUID = -8473038413268727114L;


	// P( language )
	private Map<String, Double> logdistLanguage;

	// P( prevGlyph[elisionTilde,elided,char(!elisionTilde&&!elided)] | language )
	private Map<String, Map<GlyphType, Double>> logdistGlyphType;

	// P( prevLmChar | language, prevGlyph[elisionTilde,elided,char(!elisionTilde&&!elided)] )
	private Map<String, List<List<Double>>> logdistPrevLmChar;

	// P( lmChar | language, prevGlyph[elisionTilde,elided,char(!elisionTilde&&!elided)], prevLmChar )
	private Map<String, List<List<List<Double>>>> logdistLmChar;

	// P( glyph[c1..cN,elisonTilde,elided] | language, prevGlyph[elisionTilde,elided,char(!elisionTilde&&!elided)], prevLmChar, lmChar )
	private Map<String, List<List<List<Map<GlyphChar, Double>>>>> logdistGlyph;
	
	
	public double logProb(String language, GlyphType prevGlyphChar, int prevLmChar, int lmChar, GlyphChar glyphChar) {
		double logpLang = logdistLanguage.get(language);
		double logpGlyphType = logdistGlyphType.get(language).get(prevGlyphChar);
		double logpPrevLmChar = logdistPrevLmChar.get(language).get(prevGlyphChar.ordinal()).get(prevLmChar);
		double logpLmChar = logdistLmChar.get(language).get(prevGlyphChar.ordinal()).get(prevLmChar).get(lmChar);
		double logpGlyph = logdistGlyph.get(language).get(prevGlyphChar.ordinal()).get(prevLmChar).get(lmChar).get(glyphChar);
		

		
//		// elision-tilde-marked glyphs must be followed by an elided glyph
//				if (prevGlyphChar.hasElisionTilde) {
//					if (glyphChar.isElided)
//						return 0.0; // log(1.0)
//					else
//						return Double.NEGATIVE_INFINITY; // log(0.0)
//				}
//				else {
//					// elided glyphs can only follow elision-tilde-marked glyphs or other elided glyphs
//					if (glyphChar.isElided && !prevGlyphChar.isElided) return Double.NEGATIVE_INFINITY;
//					
//					/*
//					 * For an elided glyph, we allow ourselves to look backward at the previous state's LM character, 
//					 * not just at whether the previous glyph was tilde'd or elided or not.  For any other kind of
//					 * substitution (including an elision-tilde substitution), we do not actually condition on the
//					 * previous LM char.
//					 */
//					
//					
//					
//					glyphLogProbs.get(language).get(lmChar).get(prevGlyphChar).get(glyphChar);
//				}		
		
		
		
		
		
		
		return logpLang + logpGlyphType + logpPrevLmChar + logpLmChar + logpGlyph;
	}

}
