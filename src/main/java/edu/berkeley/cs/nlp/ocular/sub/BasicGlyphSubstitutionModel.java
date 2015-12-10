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
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
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

	public BasicGlyphSubstitutionModel(double[][][][][] probs,
			Indexer<String> langIndexer, Indexer<String> charIndexer) {
		this.langIndexer = langIndexer;
		this.charIndexer = charIndexer;
		
		int numChars = charIndexer.size();
		//this.numGlyphs = numChars + 2;
		this.GLYPH_ELISION_TILDE = numChars;
		this.GLYPH_ELIDED = numChars + 1;
		this.probs = probs;
	}

	// P( glyph[c1..cN,elisonTilde,elided] | prevGlyph[elisionTilde,elided,char(!elisionTilde&&!elided)], prevLmChar, lmChar )
	public double glyphProb(int language, GlyphType prevGlyphType, int prevLmChar, int lmChar, GlyphChar glyphChar) {
		GlyphType currGlyphType = glyphChar.toGlyphType();
		int glyph = (currGlyphType == GlyphType.ELIDED ? GLYPH_ELIDED : (currGlyphType == GlyphType.ELISION_TILDE) ? GLYPH_ELISION_TILDE : glyphChar.templateCharIndex);
		return probs[language][prevGlyphType.ordinal()][prevLmChar][lmChar][glyph];
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
		private int spaceCharIndex;
		private Set<Integer> canBeReplaced;
		private Set<Integer> validSubstitutionChars;
		private Set<Integer> canBeElided;
		private Map<Integer,Integer> addTilde;
		private Map<Integer,Set<Integer>> diacriticDisregardMap;
		
		// stuff for printing out model info
		private List<Document> documents;
		private String inputPath;
		private String outputPath;

		public BasicGlyphSubstitutionModelFactory(
				double gsmSmoothingCount,
				Indexer<String> langIndexer,
				Indexer<String> charIndexer,
				String inputPath, String outputPath, List<Document> documents) {
			this.gsmSmoothingCount = gsmSmoothingCount;
			this.langIndexer = langIndexer;
			this.charIndexer = charIndexer;
			this.spaceCharIndex = charIndexer.getIndex(Charset.SPACE);
			this.canBeReplaced = makeCanBeReplacedSet(charIndexer);
			this.validSubstitutionChars = makeValidSubstitutionCharsSet(charIndexer);
			this.canBeElided = makeCanBeElidedSet(charIndexer);
			this.addTilde = makeAddTildeMap(charIndexer);
			this.diacriticDisregardMap = makeDiacriticDisregardMap(charIndexer);
			
			this.documents = documents;
			this.inputPath = inputPath;
			this.outputPath = outputPath;
		}
		
		public BasicGlyphSubstitutionModel make(List<TransitionState> fullViterbiStateSeq, CodeSwitchLanguageModel newLM, int iter, int batchId) {
			System.out.println("Estimating parameters of a new Glyph Substitution Model.  Iter: "+iter+", batch: "+batchId);

			int numLanguages = langIndexer.size();
			int numChars = charIndexer.size();
			int numGlyphTypes = GlyphType.values().length;
			int numGlyphs = numChars + 2;
			int GLYPH_ELISION_TILDE = numChars;
			int GLYPH_ELIDED = numChars + 1;
	
			//
			// Initialize the counts matrix. Add smoothing counts (and no counts for invalid options).
			//
			double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] counts = new double[numLanguages][numGlyphTypes][numChars][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				Set<Integer> langActiveChars = newLM.get(language).getActiveCharacters();
				for (GlyphType prevGlyph : GlyphType.values()) {
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
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
					//System.out.println("lang="+langIndexer.getObject(language)+"("+language+"), prevGlyphType="+prevGlyph+ ", prevLmChar="+charIndexer.getObject(prevLmChar)+"("+prevLmChar+"), lmChar="+charIndexer.getObject(lmChar)+"("+lmChar+"), glyphChar="+charIndexer.getObject(glyph)+"("+glyph+")");
				}
			}
			
			//
			// Normalize counts to get probabilities
			//
//			List<Integer> toPrintLanguage = new ArrayList<Integer>();
//			List<Integer> toPrintPrevGlyph = new ArrayList<Integer>();
//			List<Integer> toPrintPrevLmChar = new ArrayList<Integer>();
//			List<Integer> toPrintLmChar = new ArrayList<Integer>();
//			List<Integer> toPrintGlyph = new ArrayList<Integer>();
			double[/*language*/][/*prevGlyph*/][/*prevLmChar*/][/*lmChar*/][/*glyph*/] probs = new double[numLanguages][numGlyphTypes][numChars][numChars][numGlyphs];
			for (int language = 0; language < numLanguages; ++language) {
				for (GlyphType prevGlyph : GlyphType.values()) {
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							double sum = ArrayHelper.sum(counts[language][prevGlyph.ordinal()][prevLmChar][lmChar]);
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								double c = counts[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph];
								double p = (c / sum);
								probs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] = p;

								if (c > gsmSmoothingCount) {
//									System.out.println("c="+c+", lang="+langIndexer.getObject(language)+"("+language+"), prevGlyphType="+prevGlyph+ ", prevLmChar="+charIndexer.getObject(prevLmChar)+"("+prevLmChar+"), lmChar="+charIndexer.getObject(lmChar)+"("+lmChar+"), glyphChar="+charIndexer.getObject(glyph)+"("+glyph+"), p="+p+", logp="+Math.log(p));
//									toPrintLanguage.add(language);
//									toPrintPrevGlyph.add(prevGlyph.ordinal());
//									toPrintPrevLmChar.add(prevLmChar);
//									toPrintLmChar.add(lmChar);
//									toPrintGlyph.add(glyph);
								}
							}
						}
					}
				}
			}
			
			System.out.println("Writing out GSM information.");
			printGsmProbs3(numLanguages, numChars, numGlyphs, counts, probs, iter, batchId);
			//printGsmProbs2(numLanguages, numChars, numGlyphs, probs);
			//printGsmProbs(numLanguages, numChars, numGlyphs, counts, probs);
			
			return new BasicGlyphSubstitutionModel(probs, langIndexer, charIndexer);
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
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
						String sprevLmChar = charIndexer.getObject(prevLmChar);
						//if (!CHARS_TO_PRINT.contains(sprevLmChar)) continue;
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							String slmChar = charIndexer.getObject(lmChar);
							//if (!CHARS_TO_PRINT.contains(slmChar)) continue;
							
							// figure out what the lowest count is, and then exclude things with that count
							double lowCount = ArrayHelper.min(counts[language][prevGlyph.ordinal()][prevLmChar][lmChar]);
							double lowProb = ArrayHelper.min(probs[language][prevGlyph.ordinal()][prevLmChar][lmChar]);
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								//if (!(CHARS_TO_PRINT.contains(glyph) || glyph > numChars)) continue;
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
		
		//				sb.append("language\tprevGlyph\tprevLmChar\tlmChar\tglyph\t\tcount\tprob\n");
		//				for (int i = 0; i < toPrintLanguage.size(); ++i) {
		//					int glyph = toPrintGlyph.get(i);
		//					sb.append(langIndexer.getObject(toPrintLanguage.get(i))).append("\t");
		//					sb.append(toPrintPrevGlyph.get(i)).append("\t");
		//					sb.append(charIndexer.getObject(toPrintPrevLmChar.get(i))).append("\t");
		//					sb.append(charIndexer.getObject(toPrintLmChar.get(i))).append("\t");
		//					sb.append(glyph >= 0 ? charIndexer.getObject(glyph) : (glyph == numGlyphs ? "EpsilonTilde": "Elided")).append("\t\t");
		//					sb.append(counts[toPrintLanguage.get(i)][toPrintPrevGlyph.get(i)][toPrintPrevLmChar.get(i)][toPrintLmChar.get(i)][glyph] + "\t");
		//					sb.append(Math.exp(lprobs[toPrintLanguage.get(i)][toPrintPrevGlyph.get(i)][toPrintPrevLmChar.get(i)][toPrintLmChar.get(i)][glyph]) + "\t");
		//					sb.append("\n");
		//				}
			
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

		private void printGsmProbs2(int numLanguages, int numChars, int numGlyphs, double[][][][][] probs) {
			StringBuffer sb = new StringBuffer();
			sb.append("language\tprevGlyph\tprevLmChar\tlmChar\t\t"); 
			
			for (int g = 0; g < numChars; ++g) sb.append(charIndexer.getObject(g) + "\t"); 
			sb.append("EpsilonTilde\tElided\n");
			for (int language = 0; language < numLanguages; ++language) {
				String slanguage = langIndexer.getObject(language);
				for (GlyphType prevGlyph : GlyphType.values()) {
					String sprevGlyph = prevGlyph.name();
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
						String sprevLmChar = charIndexer.getObject(prevLmChar);
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							String slmChar = charIndexer.getObject(lmChar);

							sb.append(slanguage).append("\t");
							sb.append(sprevGlyph).append("\t");
							sb.append(sprevLmChar).append("\t");
							sb.append(slmChar).append("\t\t");
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								sb.append(probs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] + "\t");
							}
							sb.append("\n");
							
						}
					}
				}
			}
		}

		private void printGsmProbs(int numLanguages, int numChars, int numGlyphs, double[][][][][] counts, double[][][][][] probs) {
			StringBuffer sb = new StringBuffer();
			sb.append("language\tprevGlyph\tprevLmChar\tlmChar\t\t"); 
		
			Set<String> CHARS_TO_PRINT = setUnion(makeSet(" "), Charset.LOWERCASE_LATIN_LETTERS);
			for (String c : Charset.LOWERCASE_VOWELS) {
				CHARS_TO_PRINT.add(Charset.ACUTE_ESCAPE + c);
				CHARS_TO_PRINT.add(Charset.GRAVE_ESCAPE + c);
			}
			
			sb.append("count\t\t");
			for (int g = 0; g < numChars; ++g) { 
				if (!(CHARS_TO_PRINT.contains(g))) continue;
				sb.append(charIndexer.getObject(g) + "\t");
			}
			sb.append("EpsilonTilde\tElided\n");
			for (int language = 0; language < numLanguages; ++language) {
				String slanguage = langIndexer.getObject(language);
				for (GlyphType prevGlyph : GlyphType.values()) {
					String sprevGlyph = prevGlyph.name();
					for (int prevLmChar = 0; prevLmChar < numChars; ++prevLmChar) {
						String sprevLmChar = charIndexer.getObject(prevLmChar);
						if (!CHARS_TO_PRINT.contains(sprevLmChar)) continue;
						for (int lmChar = 0; lmChar < numChars; ++lmChar) {
							String slmChar = charIndexer.getObject(lmChar);
							if (!CHARS_TO_PRINT.contains(slmChar)) continue;
		
							sb.append(slanguage).append("\t");
							sb.append(sprevGlyph).append("\t");
							sb.append(sprevLmChar).append("\t");
							sb.append(slmChar).append("\t\t");
							sb.append(ArrayHelper.sum(counts[language][prevGlyph.ordinal()][prevLmChar][lmChar])).append("\t\t");
							for (int glyph = 0; glyph < numGlyphs; ++glyph) {
								if (!(CHARS_TO_PRINT.contains(glyph) || glyph > numChars)) continue;
								sb.append(probs[language][prevGlyph.ordinal()][prevLmChar][lmChar][glyph] + "\t");
							}
							sb.append("\n");
							
						}
					}
				}
			}
		
		//				sb.append("language\tprevGlyph\tprevLmChar\tlmChar\tglyph\t\tcount\tprob\n");
		//				for (int i = 0; i < toPrintLanguage.size(); ++i) {
		//					int glyph = toPrintGlyph.get(i);
		//					sb.append(langIndexer.getObject(toPrintLanguage.get(i))).append("\t");
		//					sb.append(toPrintPrevGlyph.get(i)).append("\t");
		//					sb.append(charIndexer.getObject(toPrintPrevLmChar.get(i))).append("\t");
		//					sb.append(charIndexer.getObject(toPrintLmChar.get(i))).append("\t");
		//					sb.append(glyph >= 0 ? charIndexer.getObject(glyph) : (glyph == numGlyphs ? "EpsilonTilde": "Elided")).append("\t\t");
		//					sb.append(counts[toPrintLanguage.get(i)][toPrintPrevGlyph.get(i)][toPrintPrevLmChar.get(i)][toPrintLmChar.get(i)][glyph] + "\t");
		//					sb.append(Math.exp(lprobs[toPrintLanguage.get(i)][toPrintPrevGlyph.get(i)][toPrintPrevLmChar.get(i)][toPrintLmChar.get(i)][glyph]) + "\t");
		//					sb.append("\n");
		//				}
			
			String fn = "";//outputPath "/u/dhg/temp/retrained-gsm-stats-iter-"+iter+".tsv";
			System.out.println("Write GSM info out to ["+fn+"]");
			FileHelper.writeString(fn, sb.toString());
		}
	}
	
}
