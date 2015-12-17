package edu.berkeley.cs.nlp.ocular.sub;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeAddTildeMap;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeCanBeElidedSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeCanBeReplacedSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeDiacriticDisregardMap;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeValidSubstitutionCharsSet;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeSet;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.setUnion;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.FileUtil;
import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.util.ArrayHelper;
import edu.berkeley.cs.nlp.ocular.util.FileHelper;
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

	private Indexer<String> langIndexer;
	private Indexer<String> charIndexer;
	private int spaceCharIndex;
	private int numChars;

	private double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] probs;
	private double gsmPower;

	public BasicGlyphSubstitutionModel(double[][][][][] probs,
			double gsmPower,
			Indexer<String> langIndexer, Indexer<String> charIndexer) {
		this.langIndexer = langIndexer;
		this.charIndexer = charIndexer;
		this.spaceCharIndex = charIndexer.getIndex(Charset.SPACE);
		this.numChars = charIndexer.size();
		
		this.probs = probs;
		this.gsmPower = gsmPower;
	}

	// P( glyph[c1..cN,elisonTilde,elided] | prevGlyph[elisionTilde,elided,char(!elisionTilde&&!elided)], prevLmChar, lmChar )
	public double glyphProb(int language, GlyphType prevGlyphType, int prevLmChar, int lmChar, GlyphChar glyphChar) {
		int prevLmCharForLookup = (glyphChar.isElided() ? prevLmChar : spaceCharIndex); // don't actually condition on the prev lm char unless we're trying to elide the current char

		GlyphType glyphType = glyphChar.toGlyphType();
		int glyph = (glyphType == GlyphType.NORMAL_CHAR) ? glyphChar.templateCharIndex : (numChars + glyphType.ordinal());
		double p = probs[language][prevGlyphType.ordinal()][prevLmCharForLookup][lmChar][glyph];
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
		private int spaceCharIndex;
		private Set<Integer> canBeReplaced;
		private Set<Integer> validSubstitutionChars;
		private Set<Integer> canBeElided;
		private Map<Integer,Integer> addTilde;
		private Map<Integer,Set<Integer>> diacriticDisregardMap;
		
		private int numLanguages;
		private int numChars;
		private int numGlyphTypes;
		private int numGlyphs;
		public int GLYPH_ELISION_TILDE;
		public int GLYPH_TILDE_ELIDED;
		public int GLYPH_FIRST_ELIDED;
		
		private double gsmPower;
		private int minCountsForEvalGsm;
		
		// stuff for printing out model info
		private List<Document> documents;
		private List<Document> evalDocuments;
		private String inputPath;
		private String outputPath;
		
		public BasicGlyphSubstitutionModelFactory(
				double gsmSmoothingCount,
				double elisionSmoothingCountMultiplier,
				Indexer<String> langIndexer,
				Indexer<String> charIndexer,
				Set<Integer>[] activeCharacterSets,
				double gsmPower, int minCountsForEvalGsm,
				String inputPath, String outputPath, List<Document> documents, List<Document> evalDocuments) {
			this.gsmSmoothingCount = gsmSmoothingCount;
			this.elisionSmoothingCountMultiplier = elisionSmoothingCountMultiplier;
			this.langIndexer = langIndexer;
			this.charIndexer = charIndexer;
			this.activeCharacterSets = activeCharacterSets;
			this.gsmPower = gsmPower;
			this.minCountsForEvalGsm = minCountsForEvalGsm;
			
			this.spaceCharIndex = charIndexer.getIndex(Charset.SPACE);
			this.canBeReplaced = makeCanBeReplacedSet(charIndexer);
			this.validSubstitutionChars = makeValidSubstitutionCharsSet(charIndexer);
			this.canBeElided = makeCanBeElidedSet(charIndexer);
			this.addTilde = makeAddTildeMap(charIndexer);
			this.diacriticDisregardMap = makeDiacriticDisregardMap(charIndexer);
			
			this.numLanguages = langIndexer.size();
			this.numChars = charIndexer.size();
			this.numGlyphTypes = GlyphType.values().length;
			this.numGlyphs = numChars + GlyphType.values().length-1;
			this.GLYPH_ELISION_TILDE = numChars + GlyphType.ELISION_TILDE.ordinal();
			this.GLYPH_TILDE_ELIDED = numChars + GlyphType.TILDE_ELIDED.ordinal();
			this.GLYPH_FIRST_ELIDED = numChars + GlyphType.FIRST_ELIDED.ordinal();
			
			this.documents = documents;
			this.evalDocuments = evalDocuments;
			this.inputPath = inputPath;
			this.outputPath = outputPath;
		}
		
		public BasicGlyphSubstitutionModel uniform() {
			return make(initializeNewCountsMatrix(), 0, 0);
		}
		
		/**
		 * Initialize the counts matrix. Add smoothing counts (and no counts for invalid options).
		 */
		public double[][][][][] initializeNewCountsMatrix() {
			double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] counts = new double[numLanguages][numGlyphTypes][numChars][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				for (GlyphType prevGlyph : GlyphType.values()) {
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = getSmoothingValue(language, prevGlyph, prevLmChar, lmChar, glyph);
							}
						}
					}
				}
			}
			return counts;
		}
		
		private boolean isElided(int glyph) {
			return glyph == GLYPH_TILDE_ELIDED || glyph == GLYPH_FIRST_ELIDED;
		}
		
		public double getSmoothingValue(int language, GlyphType prevGlyph, int prevLmChar, int lmChar, int glyph) {
			if (!activeCharacterSets[language].contains(lmChar)) return 0.0;                          // lm char must be valid for the language
			
			if (isElided(glyph) && !canBeElided.contains(lmChar)) return 0.0;       // an elided char must be elidable
			if (!isElided(glyph) && prevLmChar != spaceCharIndex) return 0.0;       // unless we are trying to elide the current char, the previous char must be marked as a "space" since we don't want to actually condition on it. 

			if (prevGlyph == GlyphType.ELISION_TILDE && glyph != GLYPH_TILDE_ELIDED) return 0.0;   // an elision-tilde-decorated char must be followed by an elision
			if (glyph == GLYPH_TILDE_ELIDED && !(prevGlyph == GlyphType.ELISION_TILDE || prevGlyph == GlyphType.TILDE_ELIDED)) return 0.0;   // an elision must be preceded by an elision-tilde-decorated char
			if (prevGlyph == GlyphType.NORMAL_CHAR && glyph == GLYPH_TILDE_ELIDED) return 0.0;     // a normal char may not be followed by an elision
			if (glyph == GLYPH_FIRST_ELIDED && !(prevGlyph == GlyphType.NORMAL_CHAR && prevLmChar == spaceCharIndex)) return 0.0; // for a glyph to be first_elided, it must come after a normal space char
			// elided chars can be followed by anything

			if (glyph == GLYPH_ELISION_TILDE && addTilde.get(lmChar) == null) return 0.0;             // an elision-tilde-decorated char must be elision-tilde-decoratable
			if (prevGlyph == GlyphType.ELISION_TILDE && addTilde.get(prevLmChar) == null) return 0.0; // a previous elision-tilde-decorated char must be elision-tilde-decoratable
			
			if (glyph < numChars && prevLmChar != spaceCharIndex) return 0.0;                         // if trying to emit a normal char, then do not condition on the prev lm char
			if (glyph == GLYPH_ELISION_TILDE && prevLmChar != spaceCharIndex) return 0.0;             // if trying to emit an elision-tilde char, then do not condition on the prev lm char
			// elided chars are conditioned on the previous lm char
			
			if (prevGlyph == GlyphType.TILDE_ELIDED && glyph == GLYPH_TILDE_ELIDED && !canBeElided.contains(prevLmChar)) return 0.0;  // an elided previous char must be elidable, if we are trying to elide the current char (since we are conditioning on the actual character)
			if (prevGlyph == GlyphType.TILDE_ELIDED && glyph != GLYPH_TILDE_ELIDED && prevLmChar != spaceCharIndex) return 0.0;       //     ... otherwise the previous state must be marked as a "space" since we don't want to condition on the actual character
			//if (prevGlyph == GlyphType.FIRST_ELIDED && !canBeElided.contains(prevLmChar)) return 0.0;  // an first-elided previous char must be elidable (it can't be followed by another elision)

			if (glyph == GLYPH_ELISION_TILDE) {
				return gsmSmoothingCount * elisionSmoothingCountMultiplier;
			}
			else if (glyph == GLYPH_TILDE_ELIDED) {
				return gsmSmoothingCount * elisionSmoothingCountMultiplier;
			}
			else if (glyph == GLYPH_FIRST_ELIDED) {
				return gsmSmoothingCount * elisionSmoothingCountMultiplier;
			}
			else { // glyph is a normal character
				Set<Integer> diacriticDisregardSet = diacriticDisregardMap.get(lmChar);
				if (diacriticDisregardSet != null && diacriticDisregardSet.contains(glyph))
					return gsmSmoothingCount * elisionSmoothingCountMultiplier;
				else if (canBeReplaced.contains(lmChar) && validSubstitutionChars.contains(glyph))
					return gsmSmoothingCount;
				else if (lmChar == glyph)
					return gsmSmoothingCount;
			}
			
			return 0.0;
		}
		
		/**
		 * Traverse the sequence of viterbi states, adding counts
		 */
		public void incrementCounts(double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] counts, List<TransitionState> fullViterbiStateSeq) {
			for (int i = 0; i < fullViterbiStateSeq.size(); ++i) {
				TransitionState prevTs = ((i > 0) ? fullViterbiStateSeq.get(i-1) : null);
				TransitionState currTs = fullViterbiStateSeq.get(i);
				
				int language = currTs.getLanguageIndex();
				if (language >= 0) {
					GlyphType prevGlyph = (prevTs != null ? prevTs.getGlyphChar().toGlyphType() : GlyphType.NORMAL_CHAR);
					GlyphChar currGlyphChar = currTs.getGlyphChar();
					int lmChar = currTs.getLmCharIndex();
					
					int prevLmChar;
					if (prevTs == null) prevLmChar = spaceCharIndex;
					else if (currGlyphChar.glyphType == GlyphType.TILDE_ELIDED) prevLmChar = prevTs.getLmCharIndex(); // the only time we care about the prev lm char is when we are deciding to elide
					else prevLmChar = spaceCharIndex;
					
					GlyphType currGlyphType = currGlyphChar.toGlyphType();
					int glyph = (currGlyphType == GlyphType.NORMAL_CHAR) ? currGlyphChar.templateCharIndex : (numChars + currGlyphType.ordinal());
					
					if (counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] < gsmSmoothingCount-1e-9) 
						throw new RuntimeException("Illegal state found in viterbi decoding result:  "
								+ "language="+langIndexer.getObject(language) + 
								", prevGlyph="+prevGlyph + 
								", prevLmChar="+charIndexer.getObject(prevLmChar) + 
								", lmChar="+charIndexer.getObject(lmChar) + 
								", glyph="+(currGlyphType == GlyphType.NORMAL_CHAR ? charIndexer.getObject(currGlyphChar.templateCharIndex) : currGlyphType));
					counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] += 1;
					//System.out.println("lang="+langIndexer.getObject(language)+"("+language+"), prevGlyphType="+prevGlyph+ ", prevLmChar="+charIndexer.getObject(prevLmChar)+"("+prevLmChar+"), lmChar="+charIndexer.getObject(lmChar)+"("+lmChar+"), glyphChar="+charIndexer.getObject(glyph)+"("+glyph+")");
				}
			}
		}

		public BasicGlyphSubstitutionModel make(double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] counts, int iter, int batchId) {
			System.out.println("Estimating parameters of a new Glyph Substitution Model.  Iter: "+iter+", batch: "+batchId);
			//
			// Normalize counts to get probabilities
			//
			double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] probs = new double[numLanguages][numGlyphTypes][numChars][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				for (GlyphType prevGlyph : GlyphType.values()) {
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							double sum = ArrayHelper.sum(counts[language][prevGlyph.ordinal()][prevLmChar][lmChar]);
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								double c = counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph];
								double p = (c > 1e-9 ? (c / sum) : 0.0);
								probs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = p;
							}
						}
					}
				}
			}
			
			System.out.println("Writing out GSM information.");
			synchronized (this) { printGsmProbs3(numLanguages, numChars, numGlyphs, counts, probs, iter, batchId, documents.get(0)); }
			
			return new BasicGlyphSubstitutionModel(probs, gsmPower, langIndexer, charIndexer);
		}

		public BasicGlyphSubstitutionModel makeForEval(double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] counts, int iter, int batchId) {
			if (evalDocuments == null) return null;
			
			double[][][][][] evalCounts = new double[numLanguages][numGlyphTypes][numChars][numChars][numGlyphs];
			
			//
			// Normalize counts to get probabilities
			//
			double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] probs = new double[numLanguages][numGlyphTypes][numChars][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				for (GlyphType prevGlyph : GlyphType.values()) {
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								double trueCount = counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] - gsmSmoothingCount;
								if (trueCount < 1e-9)
									evalCounts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = 0;
								else if (trueCount < minCountsForEvalGsm-1e-9)
									evalCounts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = 0;
								else
									evalCounts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = trueCount;
							}
							
							double sum = ArrayHelper.sum(evalCounts[language][prevGlyph.ordinal()][prevLmChar][lmChar]);
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								double c = evalCounts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph];
								double p = (c > 1e-9 ? (c / sum) : 0.0);
								probs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = p;
							}
						}
					}
				}
			}
			
			System.out.println("Writing out GSM information.");
			synchronized (this) { printGsmProbs3(numLanguages, numChars, numGlyphs, counts, probs, iter, batchId, evalDocuments.get(0)); }

			return new BasicGlyphSubstitutionModel(probs, gsmPower, langIndexer, charIndexer);
		}

		private void printGsmProbs3(int numLanguages, int numChars, int numGlyphs, double[][][][][] counts, double[][][][][] probs, int iter, int batchId, Document doc) {
			Set<String> CHARS_TO_PRINT = setUnion(makeSet(" "), Charset.LOWERCASE_LATIN_LETTERS);
//			for (String c : Charset.LOWERCASE_VOWELS) {
//				CHARS_TO_PRINT.add(Charset.ACUTE_ESCAPE + c);
//				CHARS_TO_PRINT.add(Charset.GRAVE_ESCAPE + c);
//			}
			
			StringBuffer sb = new StringBuffer();
			sb.append("language\tprevGlyph\tprevLmChar\tlmChar\tglyph\tcount\tminProb\tprob\n"); 
			for (int language = 0; language < numLanguages; ++language) {
				String slanguage = langIndexer.getObject(language);
				for (GlyphType prevGlyph : GlyphType.values()) {
					String sprevGlyph = prevGlyph.name();
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
						String sprevLmChar = charIndexer.getObject(prevLmChar);
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							String slmChar = charIndexer.getObject(lmChar);
							
							// figure out what the lowest count is, and then exclude things with that count
							double lowProb = ArrayHelper.min(probs[language][prevGlyph.ordinal()][prevLmChar][lmChar]);
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								String sglyph = glyph < numChars ? charIndexer.getObject(glyph) : GlyphType.values()[glyph-numChars].toString();
								
								double p = probs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph];
								double c = counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph];
								if (c > gsmSmoothingCount || (CHARS_TO_PRINT.contains(slmChar) && (CHARS_TO_PRINT.contains(sglyph) || glyph >= numChars))) {
									//System.out.println("c="+c+", lang="+langIndexer.getObject(language)+"("+language+"), prevGlyphType="+prevGlyph+ ", prevLmChar="+charIndexer.getObject(prevLmChar)+"("+prevLmChar+"), lmChar="+charIndexer.getObject(lmChar)+"("+lmChar+"), glyphChar="+(glyph < numChars ? charIndexer.getObject(glyph) : (glyph == numGlyphs ? "EpsilonTilde": "Elided"))+"("+glyph+"), p="+p+", logp="+Math.log(p));
									sb.append(slanguage).append("\t");
									sb.append(sprevGlyph).append("\t");
									sb.append(sprevLmChar).append("\t");
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
				}
			}
		
			String fileParent = FileUtil.removeCommonPathPrefixOfParents(new File(inputPath), new File(doc.baseName()))._2;
			String preext = "newGSM";
			String outputFilenameBase = outputPath + "/" + fileParent + "/" + preext;
			if (iter > 0) outputFilenameBase += "_iter-" + iter;
			if (batchId > 0) outputFilenameBase += "_batch-" + batchId;
			String outputFilename = outputFilenameBase + ".tsv";

			System.out.println("Writing info about newly-trained GSM on iteration "+iter+", batch "+batchId+" out to ["+outputFilename+"]");
			FileHelper.writeString(outputFilename, sb.toString());
		}
	}

}
