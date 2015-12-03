package edu.berkeley.cs.nlp.ocular.sub;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeAddTildeMap;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeCanBeElidedSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeCanBeReplacedSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeValidSubstitutionCharsSet;

import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.util.ArrayHelper;
import fileio.f;
import indexer.Indexer;

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
 * P( glyph[c,ti,el] | language, lmChar, prevLmChar, prevGlyph[pc,pti,pel] ) =
 *
 * 
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class BasicGlyphSubstitutionModel implements GlyphSubstitutionModel {
	private static final long serialVersionUID = -8473038413268727114L;

	private CodeSwitchLanguageModel newLM;
	private Indexer<String> langIndexer;
	private Indexer<String> charIndexer;
	private Set<Integer> canBeReplaced;
	private Set<Integer> validSubstitutionChars;
	private Set<Integer> canBeElided;
	private Map<Integer,Integer> addTilde;

	//private int numGlyphs;
	private int GLYPH_ELISION_TILDE;
	private int GLYPH_ELIDED;
	
	private double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] logProbs;

	public BasicGlyphSubstitutionModel(double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] logProbs,
			CodeSwitchLanguageModel newLM,
			Indexer<String> langIndexer, Indexer<String> charIndexer) {
		this.newLM = newLM;
		this.logProbs = logProbs;
		this.langIndexer = langIndexer;
		this.charIndexer = charIndexer;
		int numChars = charIndexer.size();
		//this.numGlyphs = numChars + 2;
		this.GLYPH_ELISION_TILDE = numChars;
		this.GLYPH_ELIDED = numChars + 1;
	}

	// P( glyph[c1..cN,elisonTilde,elided] | prevGlyph[elisionTilde,elided,char(!elisionTilde&&!elided)], prevLmChar, lmChar )
	public double logGlyphProb(int language, GlyphType prevGlyphType, int prevLmChar, int lmChar, GlyphChar glyphChar) {
		GlyphType currGlyphType = glyphChar.toGlyphType();
		int glyph = (currGlyphType == GlyphType.ELIDED ? GLYPH_ELIDED : (currGlyphType == GlyphType.ELISION_TILDE) ? GLYPH_ELISION_TILDE : glyphChar.templateCharIndex);
		
		double lp = logProbs[language][prevGlyphType.ordinal()][prevLmChar][lmChar][glyph];
		
		boolean invalid = true;
		Set<Integer> langActiveChars = newLM.get(language).getActiveCharacters();
		if (langActiveChars.contains(lmChar)) {
			if (glyph == GLYPH_ELISION_TILDE) {
				if (prevGlyphType != GlyphType.ELISION_TILDE) {
					if (addTilde.get(lmChar) != null) {
						invalid = false;
					}
				}
			}
			else if (glyph == GLYPH_ELIDED) {
				if (prevGlyphType != GlyphType.NORMAL_CHAR) {
					if (canBeElided.contains(lmChar)) {
						invalid = false;
					}
				}
			}
			else { // glyph is a normal character
				if (langActiveChars.contains(glyph)) {
					if (prevGlyphType != GlyphType.ELISION_TILDE) {
						if (lmChar == glyph) {
							invalid = false;
						}
						else if (canBeReplaced.contains(lmChar) && validSubstitutionChars.contains(glyph)) {
							invalid = false;
						}
					}
				}
			}
		}
		if (invalid != (lp == Double.NEGATIVE_INFINITY)) throw new RuntimeException("invalid="+invalid+", lp="+lp);
		
		return lp;
	}

	public Indexer<String> getLanguageIndexer() {
		return langIndexer;
	}

	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}

	
	public static class BasicGlyphSubstitutionModelFactory {
		private Indexer<String> langIndexer;
		private Indexer<String> charIndexer;
		private Set<Integer> canBeReplaced;
		private Set<Integer> validSubstitutionChars;
		private Set<Integer> canBeElided;
		private Map<Integer,Integer> addTilde;

		public BasicGlyphSubstitutionModelFactory(
				Indexer<String> langIndexer,
				Indexer<String> charIndexer) {
			this.langIndexer = langIndexer;
			this.charIndexer = charIndexer;
			this.canBeReplaced = makeCanBeReplacedSet(charIndexer);
			this.validSubstitutionChars = makeValidSubstitutionCharsSet(charIndexer);
			this.canBeElided = makeCanBeElidedSet(charIndexer);
			this.addTilde = makeAddTildeMap(charIndexer);
		}
		
		public BasicGlyphSubstitutionModel make(List<TransitionState> fullViterbiStateSeq, CodeSwitchLanguageModel newLM) {
			return make(fullViterbiStateSeq, newLM, null);
		}
		
		public BasicGlyphSubstitutionModel make(List<TransitionState> fullViterbiStateSeq, CodeSwitchLanguageModel newLM, Integer iter) {
			int numLanguages = langIndexer.size();
			int numChars = charIndexer.size();
			int numGlyphTypes = GlyphType.values().length;
			int numGlyphs = numChars + 2;
			int GLYPH_ELISION_TILDE = numChars;
			int GLYPH_ELIDED = numChars + 1;
	
			//
			// Initialize the counts matrix. Add smoothing counts (and no counts for invalid options).
			//
			int[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] counts = new int[numLanguages][numGlyphTypes][numChars][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				//counts[language] = new int[numGlyphTypes][numChars][numChars][numGlyphs];
				Set<Integer> langActiveChars = newLM.get(language).getActiveCharacters();
				for (GlyphType prevGlyph : GlyphType.values()) {
					//counts[language][prevGlyph.ordinal()] = new int[numChars][numChars][numGlyphs];
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
						//counts[language][prevGlyph.ordinal()][prevLmChar] = new int[numChars][numGlyphs];
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							//counts[language][prevGlyph.ordinal()][prevLmChar][lmChar] = new int[numGlyphs];
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
	
								if (langActiveChars.contains(lmChar)) {
									if (glyph == GLYPH_ELISION_TILDE) {
										if (prevGlyph != GlyphType.ELISION_TILDE) {
											if (addTilde.get(lmChar) != null) {
												counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = 1;
											}
										}
									}
									else if (glyph == GLYPH_ELIDED) {
										if (prevGlyph != GlyphType.NORMAL_CHAR) {
											if (canBeElided.contains(lmChar)) {
												counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = 1;
											}
										}
									}
									else { // glyph is a normal character
										if (langActiveChars.contains(glyph)) {
											if (prevGlyph != GlyphType.ELISION_TILDE) {
												if (lmChar == glyph) {
													counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = 1;
												}
												else if (canBeReplaced.contains(lmChar) && validSubstitutionChars.contains(glyph)) {
													counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = 1;
												}
											}
										}
									}
								}
								
							}
						}
					}
				}
			}
			
			//
			// Traverse the sequence of viterbi states, adding counts
			//
			for (int i = 0; i < fullViterbiStateSeq.size(); ++i) {
				TransitionState prevTs = fullViterbiStateSeq.get(i-1);
				TransitionState currTs = fullViterbiStateSeq.get(i);
				
				int language = currTs.getLanguageIndex();
				if (language >= 0) {
					GlyphType prevGlyph = ((i >= 0) ? prevTs.getGlyphChar().toGlyphType() : GlyphType.NORMAL_CHAR);
					int prevLmChar = prevTs.getLmCharIndex();
					int lmChar = currTs.getLmCharIndex();
					
					GlyphChar currGlyphChar = currTs.getGlyphChar();
					GlyphType currGlyphType = currGlyphChar.toGlyphType();
					int glyph = (currGlyphType == GlyphType.ELIDED ? GLYPH_ELIDED : (currGlyphType == GlyphType.ELISION_TILDE) ? GLYPH_ELISION_TILDE : currGlyphChar.templateCharIndex);
					
					if (counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] > 0) throw new RuntimeException("Illegal state found in viterbi decoding result:  language="+language + ", prevGlyph="+prevGlyph + ", prevLmChar="+prevLmChar + ", lmChar="+lmChar + ", glyph="+glyph);
					counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] += 1;
				}
			}
			
			//
			// Normalize counts to get probabilities
			//
			double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] lprobs = new double[numLanguages][numGlyphTypes][numChars][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				//lprobs[language] = new double[numGlyphTypes][numChars][numChars][numGlyphs];
				for (GlyphType prevGlyph : GlyphType.values()) {
					//lprobs[language][prevGlyph.ordinal()] = new double[numChars][numChars][numGlyphs];
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
						//lprobs[language][prevGlyph.ordinal()][prevLmChar] = new double[numChars][numGlyphs];
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							int sum = ArrayHelper.sum(counts[language][prevGlyph.ordinal()][prevLmChar][lmChar]);
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								int c = counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph];
								lprobs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = Math.log(((double)c) / sum);
							}
						}
					}
				}
			}
			
			if (iter != null) {
				StringBuilder sb = new StringBuilder();
				sb.append("language\tprevGlyph\tprevLmChar\tlmChar\t\t"); for (String g : charIndexer.getObjects()) sb.append(g + "\t"); sb.append("EpsilonTilde\tElided\n");
				
				for (int language = 0; language < numLanguages; ++language) {
					String slanguage = langIndexer.getObject(language);
					sb.append(slanguage).append("\t");
					for (GlyphType prevGlyph : GlyphType.values()) {
						String sprevGlyph = prevGlyph.name();
						sb.append(sprevGlyph).append("\t");
						for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
							String sprevLmChar = charIndexer.getObject(prevLmChar);
							sb.append(sprevLmChar).append("\t");
							for (int lmChar = 0; lmChar < numChars; ++lmChar) {
								String slmChar = charIndexer.getObject(lmChar);
								sb.append(slmChar).append("\t\t");
								for (int glyph = 0; glyph < numGlyphs; ++glyph) {
									sb.append(lprobs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] + "\t");
								}
								sb.append("\n");
							}
						}
					}
				}

				f.writeString(System.getenv("HOME")+"/temp/retrained-gsm-stats-iter-"+iter+".tsv", sb.toString());
			}
			
			return new BasicGlyphSubstitutionModel(lprobs, newLM, langIndexer, charIndexer);
		}
	}
	
}
