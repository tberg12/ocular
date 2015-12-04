package edu.berkeley.cs.nlp.ocular.sub;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeAddTildeMap;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeCanBeElidedSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeCanBeReplacedSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeValidSubstitutionCharsSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
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

	public BasicGlyphSubstitutionModel(double[][][][][] logProbs,
			CodeSwitchLanguageModel newLM,
			Indexer<String> langIndexer, Indexer<String> charIndexer, 
			Set<Integer> canBeReplaced, Set<Integer> validSubstitutionChars, Set<Integer> canBeElided, Map<Integer, Integer> addTilde) {
		this.newLM = newLM;
		this.langIndexer = langIndexer;
		this.charIndexer = charIndexer;
		
		this.canBeReplaced = canBeReplaced;
		this.validSubstitutionChars = validSubstitutionChars;
		this.canBeElided = canBeElided;
		this.addTilde = addTilde;
		
		int numChars = charIndexer.size();
		//this.numGlyphs = numChars + 2;
		this.GLYPH_ELISION_TILDE = numChars;
		this.GLYPH_ELIDED = numChars + 1;
		this.logProbs = logProbs;
	}

//	public BasicGlyphSubstitutionModel(double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] logProbs,
//			CodeSwitchLanguageModel newLM,
//			Indexer<String> langIndexer, Indexer<String> charIndexer) {
//		this.newLM = newLM;
//		this.logProbs = logProbs;
//		this.langIndexer = langIndexer;
//		this.charIndexer = charIndexer;
//	}
	
	

	// P( glyph[c1..cN,elisonTilde,elided] | prevGlyph[elisionTilde,elided,char(!elisionTilde&&!elided)], prevLmChar, lmChar )
	public double logGlyphProb(int language, GlyphType prevGlyphType, int prevLmChar, int lmChar, GlyphChar glyphChar) {
		GlyphType currGlyphType = glyphChar.toGlyphType();
		int glyph = (currGlyphType == GlyphType.ELIDED ? GLYPH_ELIDED : (currGlyphType == GlyphType.ELISION_TILDE) ? GLYPH_ELISION_TILDE : glyphChar.templateCharIndex);
		
		double lp = logProbs[language][prevGlyphType.ordinal()][prevLmChar][lmChar][glyph];
		
//		boolean invalid = true;
//		Set<Integer> langActiveChars = newLM.get(language).getActiveCharacters();
//		if (langActiveChars.contains(lmChar)) {
//			if (glyph == GLYPH_ELISION_TILDE) {
//				if (prevGlyphType != GlyphType.ELISION_TILDE) {
//					if (addTilde.get(lmChar) != null) {
//						invalid = false;
//					}
//				}
//			}
//			else if (glyph == GLYPH_ELIDED) {
//				if (prevGlyphType != GlyphType.NORMAL_CHAR) {
//					if (canBeElided.contains(lmChar)) {
//						invalid = false;
//					}
//				}
//			}
//			else { // glyph is a normal character
//				if (langActiveChars.contains(glyph)) {
//					if (prevGlyphType != GlyphType.ELISION_TILDE) {
//						if (lmChar == glyph) {
//							invalid = false;
//						}
//						else if (canBeReplaced.contains(lmChar) && validSubstitutionChars.contains(glyph)) {
//							invalid = false;
//						}
//					}
//				}
//			}
//		}
//		
//		if (invalid != (lp == Double.NEGATIVE_INFINITY)) {
//			throw new RuntimeException("invalid="+invalid+", lp="+lp+"\n  lang="+langIndexer.getObject(language)+"("+language+"), prevGlyphType="+prevGlyphType + ", prevLmChar="+charIndexer.getObject(prevLmChar)+"("+prevLmChar+"), lmChar="+charIndexer.getObject(lmChar)+"("+lmChar+"), glyphChar="+glyphChar.toString(charIndexer));
//		}
		
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
		private int spaceCharIndex;
		private Set<Integer> canBeReplaced;
		private Set<Integer> validSubstitutionChars;
		private Set<Integer> canBeElided;
		private Map<Integer,Integer> addTilde;

		public BasicGlyphSubstitutionModelFactory(
				Indexer<String> langIndexer,
				Indexer<String> charIndexer) {
			this.langIndexer = langIndexer;
			this.charIndexer = charIndexer;
			this.spaceCharIndex = charIndexer.getIndex(Charset.SPACE);
			this.canBeReplaced = makeCanBeReplacedSet(charIndexer);
			this.validSubstitutionChars = makeValidSubstitutionCharsSet(charIndexer);
			this.canBeElided = makeCanBeElidedSet(charIndexer);
			this.addTilde = makeAddTildeMap(charIndexer);
		}
		
		public BasicGlyphSubstitutionModel make(List<TransitionState> fullViterbiStateSeq, double selfEmissionBias, CodeSwitchLanguageModel newLM) {
			return make(fullViterbiStateSeq, selfEmissionBias, newLM, null);
		}
		
		public BasicGlyphSubstitutionModel make(List<TransitionState> fullViterbiStateSeq, double selfEmissionBias, CodeSwitchLanguageModel newLM, Integer iter) {
			System.out.println("Estimating parameters of a new Glyph Substitution Model.  " + (iter != null ? "Iter: "+iter : ""));

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
								counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = 1;
	
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
				TransitionState prevTs = ((i > 0) ? fullViterbiStateSeq.get(i-1) : null);
				TransitionState currTs = fullViterbiStateSeq.get(i);
				
				int language = currTs.getLanguageIndex();
				if (language >= 0) {
					GlyphType prevGlyph = (prevTs != null ? prevTs.getGlyphChar().toGlyphType() : GlyphType.NORMAL_CHAR);
					int prevLmChar = (prevTs != null ? prevTs.getLmCharIndex() : spaceCharIndex);
					int lmChar = currTs.getLmCharIndex();
					
					GlyphChar currGlyphChar = currTs.getGlyphChar();
					GlyphType currGlyphType = currGlyphChar.toGlyphType();
					int glyph = (currGlyphType == GlyphType.ELIDED ? GLYPH_ELIDED : (currGlyphType == GlyphType.ELISION_TILDE) ? GLYPH_ELISION_TILDE : currGlyphChar.templateCharIndex);
					
					if (counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] < 1) throw new RuntimeException("Illegal state found in viterbi decoding result:  language="+language + ", prevGlyph="+prevGlyph + ", prevLmChar="+prevLmChar + ", lmChar="+lmChar + ", glyph="+glyph);
					counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] += 1;
				}
			}
			
			//
			// Normalize counts to get probabilities
			//
			List<Integer> toPrintLanguage = new ArrayList<Integer>();
			List<Integer> toPrintPrevGlyph = new ArrayList<Integer>();
			List<Integer> toPrintPrevLmChar = new ArrayList<Integer>();
			List<Integer> toPrintLmChar = new ArrayList<Integer>();
			List<Integer> toPrintGlyph = new ArrayList<Integer>();
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
								double p = (1.0 - selfEmissionBias) * ((double)c) / sum;
								double pWithBias = (glyph == lmChar ? selfEmissionBias + p : p);
								lprobs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = Math.log(pWithBias);
								
								if (c > 1) {
									toPrintLanguage.add(language);
									toPrintPrevGlyph.add(prevGlyph.ordinal());
									toPrintPrevLmChar.add(prevLmChar);
									toPrintLmChar.add(lmChar);
									toPrintGlyph.add(glyph);
								}
							}
						}
					}
				}
			}
			
			if (iter != null) {
				System.out.println("Writing out GSM information.");
				StringBuilder sb = new StringBuilder();
				sb.append("language\tprevGlyph\tprevLmChar\tlmChar\t\t"); 
				
//				for (int g = (numChars-20); g < numChars; ++g) sb.append(charIndexer.getObject(g) + "\t"); 
//				sb.append("EpsilonTilde\tElided\n");
//				for (int language = 0; language < numLanguages; ++language) {
//					String slanguage = langIndexer.getObject(language);
//					for (GlyphType prevGlyph : GlyphType.values()) {
//						String sprevGlyph = prevGlyph.name();
//						for (int prevLmChar = (numChars-20); prevLmChar < numChars; ++prevLmChar) {
//							String sprevLmChar = charIndexer.getObject(prevLmChar);
//							for (int lmChar = (numChars-20); lmChar < numChars; ++lmChar) {
//								String slmChar = charIndexer.getObject(lmChar);
//
//								sb.append(slanguage).append("\t");
//								sb.append(sprevGlyph).append("\t");
//								sb.append(sprevLmChar).append("\t");
//								sb.append(slmChar).append("\t\t");
//								for (int glyph = (numChars-20); glyph < numGlyphs; ++glyph) {
//									sb.append(lprobs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] + "\t");
//								}
//								sb.append("\n");
//								
//							}
//						}
//					}
//				}
				
				sb.append("language\tprevGlyph\tprevLmChar\tlmChar\tglyph\t\tcount\tprob\n");
				for (int i = 0; i < toPrintLanguage.size(); ++i) {
					int glyph = toPrintGlyph.get(i);
					sb.append(langIndexer.getObject(toPrintLanguage.get(i))).append("\t");
					sb.append(toPrintPrevGlyph.get(i)).append("\t");
					sb.append(charIndexer.getObject(toPrintPrevLmChar.get(i))).append("\t");
					sb.append(charIndexer.getObject(toPrintLmChar.get(i))).append("\t");
					sb.append(glyph >= 0 ? charIndexer.getObject(glyph) : (glyph == numGlyphs ? "EpsilonTilde": "Elided")).append("\t\t");
					sb.append(counts[toPrintLanguage.get(i)][toPrintPrevGlyph.get(i)][toPrintPrevLmChar.get(i)][toPrintLmChar.get(i)][glyph] + "\t");
					sb.append(Math.exp(lprobs[toPrintLanguage.get(i)][toPrintPrevGlyph.get(i)][toPrintPrevLmChar.get(i)][toPrintLmChar.get(i)][glyph]) + "\t");
					sb.append("\n");
				}
				
				String fn = "/u/dhg/temp/retrained-gsm-stats-iter-"+iter+".tsv";
				System.out.println("Write GSM info out to ["+fn+"]");
				f.writeString(fn, sb.toString());
				//System.exit(1);
			}
			
			return new BasicGlyphSubstitutionModel(lprobs, newLM, langIndexer, charIndexer, canBeReplaced, validSubstitutionChars, canBeElided, addTilde);
		}
	}
	
}
