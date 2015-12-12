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

	//private int numGlyphs;
	private int GLYPH_ELISION_TILDE;
	private int GLYPH_ELIDED;
	
	private double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] probs;
	private boolean collapsePrevLmChar;

	public BasicGlyphSubstitutionModel(double[][][][][] probs,
			boolean collapsePrevLmChar,
			Indexer<String> langIndexer, Indexer<String> charIndexer) {
		this.langIndexer = langIndexer;
		this.charIndexer = charIndexer;
		
		int numChars = charIndexer.size();
		//this.numGlyphs = numChars + 2;
		this.GLYPH_ELISION_TILDE = numChars;
		this.GLYPH_ELIDED = numChars + 1;

		this.probs = probs;
		this.collapsePrevLmChar = collapsePrevLmChar;
	}

	// P( glyph[c1..cN,elisonTilde,elided] | prevGlyph[elisionTilde,elided,char(!elisionTilde&&!elided)], prevLmChar, lmChar )
	public double glyphProb(int language, GlyphType prevGlyphType, int prevLmChar, int lmChar, GlyphChar glyphChar) {
		GlyphType currGlyphType = glyphChar.toGlyphType();
		int glyph = (currGlyphType == GlyphType.ELIDED ? GLYPH_ELIDED : (currGlyphType == GlyphType.ELISION_TILDE) ? GLYPH_ELISION_TILDE : glyphChar.templateCharIndex);
		return probs[language][prevGlyphType.ordinal()][(collapsePrevLmChar ? 0 : prevLmChar)][lmChar][glyph];
	}

	public Indexer<String> getLanguageIndexer() {
		return langIndexer;
	}

	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}

	
	public static class BasicGlyphSubstitutionModelFactory {
		private double gsmSmoothingCount;
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
		private int GLYPH_ELISION_TILDE;
		private int GLYPH_ELIDED;
		
		private boolean collapsePrevLmChar;
		private int minCountsForEvalGsm;
		
		// stuff for printing out model info
		private List<Document> documents;
		private String inputPath;
		private String outputPath;
		
		public BasicGlyphSubstitutionModelFactory(
				double gsmSmoothingCount,
				Indexer<String> langIndexer,
				Indexer<String> charIndexer,
				Set<Integer>[] activeCharacterSets,
				boolean collapsePrevLmChar, int minCountsForEvalGsm,
				String inputPath, String outputPath, List<Document> documents) {
			this.gsmSmoothingCount = gsmSmoothingCount;
			this.langIndexer = langIndexer;
			this.charIndexer = charIndexer;
			this.activeCharacterSets = activeCharacterSets;
			this.collapsePrevLmChar = collapsePrevLmChar;
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
			this.numGlyphs = numChars + 2;
			this.GLYPH_ELISION_TILDE = numChars;
			this.GLYPH_ELIDED = numChars + 1;
			
			this.documents = documents;
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
			double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] counts = new double[numLanguages][numGlyphTypes][(collapsePrevLmChar ? 1 : numChars)][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				Set<Integer> langActiveChars = activeCharacterSets[language];
				for (GlyphType prevGlyph : GlyphType.values()) {
					for (int prevLmChar = 0; prevLmChar < (collapsePrevLmChar ? 1 : numChars); ++prevLmChar) {
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = gsmSmoothingCount;
	
								if (langActiveChars.contains(lmChar)) {
									if (glyph == GLYPH_ELISION_TILDE) {
										if (prevGlyph != GlyphType.ELISION_TILDE) {
											if (addTilde.get(lmChar) != null) {
												counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = gsmSmoothingCount;
											}
										}
									}
									else if (glyph == GLYPH_ELIDED) {
										if (prevGlyph != GlyphType.NORMAL_CHAR) {
											if (canBeElided.contains(lmChar)) {
												counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = gsmSmoothingCount;
											}
										}
									}
									else { // glyph is a normal character
										if (langActiveChars.contains(glyph)) {
											if (prevGlyph != GlyphType.ELISION_TILDE) {
												Set<Integer> diacriticDisregardSet = diacriticDisregardMap.get(lmChar);
												if (lmChar == glyph ||
													(canBeReplaced.contains(lmChar) && validSubstitutionChars.contains(glyph)) ||
													(diacriticDisregardSet != null && diacriticDisregardSet.contains(glyph))) {
														counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = gsmSmoothingCount;
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
			return counts;
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
					int prevLmChar = (collapsePrevLmChar ? 0 : (prevTs != null ? prevTs.getLmCharIndex() : spaceCharIndex));
					int lmChar = currTs.getLmCharIndex();
					
					GlyphChar currGlyphChar = currTs.getGlyphChar();
					GlyphType currGlyphType = currGlyphChar.toGlyphType();
					int glyph = (currGlyphType == GlyphType.ELIDED ? GLYPH_ELIDED : (currGlyphType == GlyphType.ELISION_TILDE) ? GLYPH_ELISION_TILDE : currGlyphChar.templateCharIndex);
					
					if (counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] < gsmSmoothingCount-1e-9) throw new RuntimeException("Illegal state found in viterbi decoding result:  language="+language + ", prevGlyph="+prevGlyph + ", prevLmChar="+prevLmChar + ", lmChar="+lmChar + ", glyph="+glyph);
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
			double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] probs = new double[numLanguages][numGlyphTypes][(collapsePrevLmChar ? 1 : numChars)][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				for (GlyphType prevGlyph : GlyphType.values()) {
					for (int prevLmChar = 0; prevLmChar < (collapsePrevLmChar ? 1 : numChars); ++prevLmChar) {
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
			synchronized (this) { printGsmProbs3(numLanguages, numChars, numGlyphs, counts, probs, iter, batchId); }
			
			return new BasicGlyphSubstitutionModel(probs, collapsePrevLmChar, langIndexer, charIndexer);
		}

		public BasicGlyphSubstitutionModel makeForEval(double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] counts) {
			double[][][][][] evalCounts = new double[numLanguages][numGlyphTypes][(collapsePrevLmChar ? 1 : numChars)][numChars][numGlyphs];
			
			//
			// Normalize counts to get probabilities
			//
			double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] probs = new double[numLanguages][numGlyphTypes][(collapsePrevLmChar ? 1 : numChars)][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				for (GlyphType prevGlyph : GlyphType.values()) {
					for (int prevLmChar = 0; prevLmChar < (collapsePrevLmChar ? 1 : numChars); ++prevLmChar) {
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								if (counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] - gsmSmoothingCount < 1e-9)
									evalCounts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = 0;
								else if (counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] - gsmSmoothingCount < minCountsForEvalGsm-1e-9)
									evalCounts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = 0;
								else 
									evalCounts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph];
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
			
			return new BasicGlyphSubstitutionModel(probs, collapsePrevLmChar, langIndexer, charIndexer);
		}

		private void printGsmProbs3(int numLanguages, int numChars, int numGlyphs, double[][][][][] counts, double[][][][][] probs, int iter, int batchId) {
			Set<String> CHARS_TO_PRINT = setUnion(makeSet(" "), Charset.LOWERCASE_LATIN_LETTERS);
			for (String c : Charset.LOWERCASE_VOWELS) {
				CHARS_TO_PRINT.add(Charset.ACUTE_ESCAPE + c);
				CHARS_TO_PRINT.add(Charset.GRAVE_ESCAPE + c);
			}
			
			StringBuffer sb = new StringBuffer();
			sb.append("language\tprevGlyph\tprevLmChar\tlmChar\tglyph\tmincount\tcount\tminProb\tprob\n"); 
			for (int language = 0; language < numLanguages; ++language) {
				String slanguage = langIndexer.getObject(language);
				for (GlyphType prevGlyph : GlyphType.values()) {
					String sprevGlyph = prevGlyph.name();
					for (int prevLmChar = 0; prevLmChar < (collapsePrevLmChar ? 1 : numChars); ++prevLmChar) {
						String sprevLmChar = (collapsePrevLmChar ? " " : charIndexer.getObject(prevLmChar));
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							String slmChar = charIndexer.getObject(lmChar);
							
							// figure out what the lowest count is, and then exclude things with that count
							double lowCount = ArrayHelper.min(counts[language][prevGlyph.ordinal()][prevLmChar][lmChar]);
							double lowProb = ArrayHelper.min(probs[language][prevGlyph.ordinal()][prevLmChar][lmChar]);
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								double p = probs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph];
								double c =  counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph];
								if (c > gsmSmoothingCount) {
									//System.out.println("c="+c+", lang="+langIndexer.getObject(language)+"("+language+"), prevGlyphType="+prevGlyph+ ", prevLmChar="+charIndexer.getObject(prevLmChar)+"("+prevLmChar+"), lmChar="+charIndexer.getObject(lmChar)+"("+lmChar+"), glyphChar="+(glyph < numChars ? charIndexer.getObject(glyph) : (glyph == numGlyphs ? "EpsilonTilde": "Elided"))+"("+glyph+"), p="+p+", logp="+Math.log(p));
									sb.append(slanguage).append("\t");
									sb.append(sprevGlyph).append("\t");
									sb.append(sprevLmChar).append("\t");
									sb.append(slmChar).append("\t");
									sb.append(glyph < numChars ? charIndexer.getObject(glyph) : (glyph == numGlyphs ? "EpsilonTilde": "Elided")).append("\t");
									sb.append(lowCount).append("\t");
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
		
			Document doc = documents.get(0);
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
