package edu.berkeley.cs.nlp.ocular.gsm;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeAddTildeMap;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeCanBeElidedSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeCanBeReplacedSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeDiacriticDisregardMap;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeValidDoublableSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeValidSubstitutionCharsSet;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeSet;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.setUnion;

import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.model.DecodeState;
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.ArrayHelper;
import edu.berkeley.cs.nlp.ocular.util.FileHelper;
import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class BasicGlyphSubstitutionModel implements GlyphSubstitutionModel {
	private static final long serialVersionUID = -8473038413268727114L;

	private Indexer<String> langIndexer;
	private Indexer<String> charIndexer;

	private int numChars;

	private double[/*language*/][/*lmChar*/][/*glyph*/] probs;
	private double gsmPower;

	public BasicGlyphSubstitutionModel(double[][][] probs,
			double gsmPower,
			Indexer<String> langIndexer,
			Indexer<String> charIndexer) {
		this.langIndexer = langIndexer;
		this.charIndexer = charIndexer;
		this.numChars = charIndexer.size();
		
		this.probs = probs;
		this.gsmPower = gsmPower;
	}

	public double glyphProb(int language, int lmChar, GlyphChar glyphChar) {
		GlyphType glyphType = glyphChar.glyphType;
		int glyph = (glyphType == GlyphType.NORMAL_CHAR) ? glyphChar.templateCharIndex : (numChars + glyphType.ordinal());
		double p = probs[language][lmChar][glyph];
		return Math.pow(p, gsmPower);
	}
	
	public Indexer<String> getLanguageIndexer() {
		return langIndexer;
	}

	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}

	
	public static class BasicGlyphSubstitutionModelFactory {
		private double gsmSmoothingCount;
		private double elisionSmoothingCountMultiplier;
		private Indexer<String> langIndexer;
		private Indexer<String> charIndexer;
		private Set<Integer>[] activeCharacterSets;
		private Set<Integer> canBeReplaced;
		private Set<Integer> canBeDoubled;
		private Set<Integer> validSubstitutionChars;
		private Set<Integer> canBeElided;
		private Map<Integer,Integer> addTilde;
		private Map<Integer,Integer> diacriticDisregardMap;
		private int sCharIndex;
		private int longsCharIndex;
		private int fCharIndex;
		private int lCharIndex;
		private int hyphenCharIndex;
		private int spaceCharIndex;
		
		private int numLanguages;
		private int numChars;
		private int numGlyphs;
		public final int GLYPH_ELISION_TILDE;
		public final int GLYPH_TILDE_ELIDED;
		public final int GLYPH_FIRST_ELIDED;
		public final int GLYPH_DOUBLED;
		public final int GLYPH_ELIDED;
		//public final int GLYPH_RMRGN_HPHN_DROP;
		
		private double gsmPower;
		private int minCountsForEvalGsm;
		
		private String outputPath;
		
		public BasicGlyphSubstitutionModelFactory(
				double gsmSmoothingCount,
				double elisionSmoothingCountMultiplier,
				Indexer<String> langIndexer,
				Indexer<String> charIndexer,
				Set<Integer>[] activeCharacterSets,
				double gsmPower, int minCountsForEvalGsm,
				String outputPath) {
			this.gsmSmoothingCount = gsmSmoothingCount;
			this.elisionSmoothingCountMultiplier = elisionSmoothingCountMultiplier;
			this.langIndexer = langIndexer;
			this.charIndexer = charIndexer;
			this.activeCharacterSets = activeCharacterSets;
			this.gsmPower = gsmPower;
			this.minCountsForEvalGsm = minCountsForEvalGsm;
			
			this.canBeReplaced = makeCanBeReplacedSet(charIndexer);
			this.canBeDoubled = makeValidDoublableSet(charIndexer);
			this.validSubstitutionChars = makeValidSubstitutionCharsSet(charIndexer);
			this.canBeElided = makeCanBeElidedSet(charIndexer);
			this.addTilde = makeAddTildeMap(charIndexer);
			this.diacriticDisregardMap = makeDiacriticDisregardMap(charIndexer);
			
			this.sCharIndex = charIndexer.contains("s") ? charIndexer.getIndex("s") : -1;
			this.longsCharIndex = charIndexer.getIndex(Charset.LONG_S);
			this.fCharIndex = charIndexer.contains("f") ? charIndexer.getIndex("f") : -1;
			this.lCharIndex = charIndexer.contains("l") ? charIndexer.getIndex("l") : -1;
			this.hyphenCharIndex = charIndexer.getIndex(Charset.HYPHEN);
			this.spaceCharIndex = charIndexer.getIndex(Charset.SPACE);
			
			this.numLanguages = langIndexer.size();
			this.numChars = charIndexer.size();
			this.numGlyphs = numChars + GlyphType.values().length-1;
			this.GLYPH_ELISION_TILDE = numChars + GlyphType.ELISION_TILDE.ordinal();
			this.GLYPH_TILDE_ELIDED = numChars + GlyphType.TILDE_ELIDED.ordinal();
			this.GLYPH_FIRST_ELIDED = numChars + GlyphType.FIRST_ELIDED.ordinal();
			this.GLYPH_DOUBLED = numChars + GlyphType.DOUBLED.ordinal();
			//this.GLYPH_RMRGN_HPHN_DROP = numChars + GlyphType.RMRGN_HPHN_DROP.ordinal();
			this.GLYPH_ELIDED = numChars + GlyphType.ELIDED.ordinal();
			
			this.outputPath = outputPath;
		}
		
		public GlyphSubstitutionModel uniform() {
			return make(initializeNewCountsMatrix(), 0, 0);
		}
		
		/**
		 * Initialize the counts matrix. Add smoothing counts (and no counts for invalid options).
		 */
		public double[][][] initializeNewCountsMatrix() {
			double[/*language*/][/*lmChar*/][/*glyph*/] counts = new double[numLanguages][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				for (int lmChar = 0; lmChar < numChars; ++lmChar) {
					for (int glyph = 0; glyph < numGlyphs; ++glyph) {
						counts[language][lmChar][glyph] = getSmoothingValue(language, lmChar, glyph);
					}
				}
			}
			return counts;
		}
		
//		private boolean isElided(int glyph) {
//			return glyph == GLYPH_TILDE_ELIDED || glyph == GLYPH_FIRST_ELIDED;
//		}
		
		public double getSmoothingValue(int language, int lmChar, int glyph) {
			
//			if (glyph != GLYPH_TILDE_ELIDED && prevLmChar != spaceCharIndex) return 0.0;                                                     // unless we are trying to elide the current char, the previous char must be marked as a "space" since we don't want to actually condition on it. 

//			if (prevGlyph == GlyphType.ELISION_TILDE && glyph != GLYPH_TILDE_ELIDED) return 0.0;                                             // an elision-tilde-decorated char must be followed by an elision
//			if (glyph == GLYPH_TILDE_ELIDED && !(prevGlyph == GlyphType.ELISION_TILDE || prevGlyph == GlyphType.TILDE_ELIDED)) return 0.0;   // an elision must be preceded by an elision-tilde-decorated char
//			if (prevGlyph == GlyphType.NORMAL_CHAR && glyph == GLYPH_TILDE_ELIDED) return 0.0;                                               // a normal char may not be followed by an elision
//			if (glyph == GLYPH_FIRST_ELIDED && !(prevGlyph == GlyphType.NORMAL_CHAR && prevLmChar == spaceCharIndex)) return 0.0;            // for a glyph to be first_elided, it must come after a normal space char
			// elided chars can be followed by anything

//			if (prevGlyph == GlyphType.ELISION_TILDE && addTilde.get(prevLmChar) == null) return 0.0;                                        // a previous elision-tilde-decorated char must be elision-tilde-decoratable
			
//			if (prevGlyph == GlyphType.TILDE_ELIDED && glyph == GLYPH_TILDE_ELIDED && !canBeElided.contains(prevLmChar)) return 0.0;         // an elided previous char must be elidable, if we are trying to elide the current char (since we are conditioning on the actual character)
//			if (prevGlyph == GlyphType.TILDE_ELIDED && glyph != GLYPH_TILDE_ELIDED && prevLmChar != spaceCharIndex) return 0.0;              //     ... otherwise the previous state must be marked as a "space" since we don't want to condition on the actual character
			//if (prevGlyph == GlyphType.FIRST_ELIDED && !canBeElided.contains(prevLmChar)) return 0.0;                                      // an first-elided previous char must be elidable (it can't be followed by another elision)

			
			
			if (!(activeCharacterSets[language].contains(lmChar) || lmChar == hyphenCharIndex)) return 0.0; // lm char must be valid for the language
			
			if (glyph == GLYPH_ELISION_TILDE) {
				if (addTilde.get(lmChar) == null) return 0.0; // an elision-tilde-decorated char must be elision-tilde-decoratable
				return gsmSmoothingCount * elisionSmoothingCountMultiplier;
			}
			else if (glyph == GLYPH_TILDE_ELIDED) {
				if (!canBeElided.contains(lmChar)) return 0.0; // an elided char must be elidable
				return gsmSmoothingCount * elisionSmoothingCountMultiplier;
			}
			else if (glyph == GLYPH_FIRST_ELIDED) {
				if (!canBeElided.contains(lmChar)) return 0.0; // an elided char must be elidable
				return gsmSmoothingCount * elisionSmoothingCountMultiplier;
			}
			else if (glyph == GLYPH_DOUBLED) {
				if (!canBeDoubled.contains(lmChar)) return 0.0; // a doubled character has to be doubleable
				return gsmSmoothingCount;// * elisionSmoothingCountMultiplier;
			}
//			else if (glyph == GLYPH_RMRGN_HPHN_DROP) {
//				if (lmChar != hyphenCharIndex) return 0.0; // only a hyphen can be hyphen-dropped
//				return gsmSmoothingCount;
//			}
			else if (glyph == GLYPH_ELIDED) {
				if (!canBeElided.contains(lmChar)) return 0.0; // an elided char must be elidable
				return gsmSmoothingCount;
			}
			else { // glyph is a normal character
				Integer baseChar = diacriticDisregardMap.get(lmChar);
				if (baseChar != null && baseChar.equals(glyph))
					return gsmSmoothingCount * elisionSmoothingCountMultiplier;
				else if (lmChar == sCharIndex && glyph == longsCharIndex)
					return gsmSmoothingCount;
				else if (lmChar == sCharIndex && (glyph == fCharIndex || glyph == lCharIndex))
					return 0.0;
				else if (lmChar == hyphenCharIndex && glyph == spaceCharIndex) // so that line-break hyphens can be elided
					return gsmSmoothingCount;
				else if (canBeReplaced.contains(lmChar) && validSubstitutionChars.contains(glyph) && activeCharacterSets[language].contains(glyph))
					return gsmSmoothingCount;
				else if (lmChar == glyph)
					return gsmSmoothingCount;
				else
					return 0.0;
			}
			
		}
		
		/**
		 * Traverse the sequence of viterbi states, adding counts
		 */
		public void incrementCounts(double[/*language*/][/*lmChar*/][/*glyph*/] counts, List<DecodeState> fullViterbiStateSeq) {
			for (int i = 0; i < fullViterbiStateSeq.size(); ++i) {
				TransitionState currTs = fullViterbiStateSeq.get(i).ts;
				TransitionStateType currType = currTs.getType();
				if (currType == TransitionStateType.TMPL) {
					int language = currTs.getLanguageIndex();
					if (language >= 0) {
						int lmChar = currTs.getLmCharIndex();
						int glyph = glyphIndex(currTs.getGlyphChar());
						counts[language][lmChar][glyph] += 1;
					}
				}
				else if (currType == TransitionStateType.RMRGN_HPHN_INIT) {
					int language = currTs.getLanguageIndex();
					if (language >= 0) {
						GlyphChar currGlyphChar = currTs.getGlyphChar();
						if (currGlyphChar.templateCharIndex == spaceCharIndex) { // line-break hyphen was elided
							int glyph = glyphIndex(currGlyphChar);
							counts[language][hyphenCharIndex][glyph] += 1;
						}
					}
				}
			}
		}
		
		private int glyphIndex(GlyphChar glyphChar) {
			return glyphChar.glyphType == GlyphType.NORMAL_CHAR ? glyphChar.templateCharIndex : (numChars + glyphChar.glyphType.ordinal());
		}

		public BasicGlyphSubstitutionModel make(double[/*language*/][/*lmChar*/][/*glyph*/] counts, int iter, int batchId) {
			// Normalize counts to get probabilities
			double[/*language*/][/*lmChar*/][/*glyph*/] probs = new double[numLanguages][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
					for (int lmChar = 0; lmChar < numChars; ++lmChar) {
						double sum = ArrayHelper.sum(counts[language][lmChar]);
						for (int glyph = 0; glyph < numGlyphs; ++glyph) {
							double c = counts[language][lmChar][glyph];
							double p = (c > 1e-9 ? (c / sum) : 0.0);
							probs[language][lmChar][glyph] = p;
						}
					}
				}
			}
			
			//System.out.println("Writing out GSM information.");
			//synchronized (this) { printGsmProbs3(numLanguages, numChars, numGlyphs, counts, probs, iter, batchId, gsmPrintoutFilepath(iter, batchId)); }
			
			return new BasicGlyphSubstitutionModel(probs, gsmPower, langIndexer, charIndexer);
		}

		public BasicGlyphSubstitutionModel makeForEval(double[/*language*/][/*lmChar*/][/*glyph*/] counts, int iter, int batchId) {
			return makeForEval(counts, iter, batchId, minCountsForEvalGsm);
		}

		public BasicGlyphSubstitutionModel makeForEval(double[/*language*/][/*lmChar*/][/*glyph*/] counts, int iter, int batchId, double minCountsForEvalGsm) {
			if (minCountsForEvalGsm < 1) {
				System.out.println("Estimating parameters of a new Glyph Substitution Model.  Iter: "+iter+", batch: "+batchId);
				return make(counts, iter, batchId);
			}
			else {
				// Normalize counts to get probabilities
				double[/*language*/][/*lmChar*/][/*glyph*/] evalCounts = new double[numLanguages][numChars][numGlyphs];
				double[/*language*/][/*lmChar*/][/*glyph*/] probs = new double[numLanguages][numChars][numGlyphs];
				for (int language = 0; language < numLanguages; ++language) {
					for (int lmChar = 0; lmChar < numChars; ++lmChar) {
						
						for (int glyph = 0; glyph < numGlyphs; ++glyph) {
							double trueCount = counts[language][lmChar][glyph] - gsmSmoothingCount;
							if (trueCount < 1e-9)
								evalCounts[language][lmChar][glyph] = 0;
							else if (trueCount < minCountsForEvalGsm-1e-9)
								evalCounts[language][lmChar][glyph] = 0;
							else
								evalCounts[language][lmChar][glyph] = trueCount;
						}
						
						double sum = ArrayHelper.sum(evalCounts[language][lmChar]);
						for (int glyph = 0; glyph < numGlyphs; ++glyph) {
							double c = evalCounts[language][lmChar][glyph];
							double p = (c > 1e-9 ? (c / sum) : 0.0);
							probs[language][lmChar][glyph] = p;
						}
					}
				}
				
				//System.out.println("Writing out GSM information.");
				//synchronized (this) { printGsmProbs3(numLanguages, numChars, numGlyphs, counts, probs, iter, batchId, gsmPrintoutFilepath(iter, batchId)+"_eval"); }
	
				return new BasicGlyphSubstitutionModel(probs, gsmPower, langIndexer, charIndexer);
			}
		}

		private void printGsmProbs3(int numLanguages, int numChars, int numGlyphs, double[][][] counts, double[][][] probs, int iter, int batchId, String outputFilenameBase) {
			Set<String> CHARS_TO_PRINT = setUnion(makeSet(" ","-","a","b","c","d",Charset.LONG_S));
			StringBuffer sb = new StringBuffer();
			sb.append("language\tlmChar\tglyph\tcount\tminProb\tprob\n"); 
			for (int language = 0; language < numLanguages; ++language) {
				String slanguage = langIndexer.getObject(language);
				for (int lmChar = 0; lmChar < numChars; ++lmChar) {
					String slmChar = charIndexer.getObject(lmChar);
					
					// figure out what the lowest count is, and then exclude things with that count
					double lowProb = ArrayHelper.min(probs[language][lmChar]);
					for (int glyph = 0; glyph < numGlyphs; ++glyph) {
						String sglyph = glyph < numChars ? charIndexer.getObject(glyph) : GlyphType.values()[glyph-numChars].toString();
						
						double p = probs[language][lmChar][glyph];
						double c = counts[language][lmChar][glyph];
						if (c > gsmSmoothingCount || (CHARS_TO_PRINT.contains(slmChar) && (CHARS_TO_PRINT.contains(sglyph) || glyph >= numChars))) {
							//System.out.println("c="+c+", lang="+langIndexer.getObject(language)+"("+language+"), prevGlyphType="+prevGlyph+ ", prevLmChar="+charIndexer.getObject(prevLmChar)+"("+prevLmChar+"), lmChar="+charIndexer.getObject(lmChar)+"("+lmChar+"), glyphChar="+(glyph < numChars ? charIndexer.getObject(glyph) : (glyph == numGlyphs ? "EpsilonTilde": "Elided"))+"("+glyph+"), p="+p+", logp="+Math.log(p));
							sb.append(slanguage).append("\t");
							sb.append(slmChar).append("\t");
							sb.append(sglyph).append("\t");
							sb.append(c).append("\t");
							sb.append(lowProb).append("\t");
							sb.append(p).append("\t");
							sb.append("\n");
						}
					}
				}
			}
		
			String outputFilename = outputFilenameBase + ".tsv";
			System.out.println("Writing info about newly-trained GSM on iteration "+iter+", batch "+batchId+" out to ["+outputFilename+"]");
			FileHelper.writeString(outputFilename, sb.toString());
		}

		private String gsmPrintoutFilepath(int iter, int batchId) {
			String preext = "newGSM";
			String outputFilenameBase = outputPath + "/gsm/" + preext;
			if (iter > 0) outputFilenameBase += "_iter-" + iter;
			if (batchId > 0) outputFilenameBase += "_batch-" + batchId;
			return outputFilenameBase;
		}
	}

	public Indexer<String> getLangIndexer() { return langIndexer; }
	public Indexer<String> getCharIndexer() { return charIndexer; }
}
