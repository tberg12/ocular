package edu.berkeley.cs.nlp.ocular.model.transition;

import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import tberg.murphy.arrays.a;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.lm.FixedLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class FixedAlignTransition implements SparseTransitionModel {
	
	public class CharacterNgramTransitionState implements SparseTransitionModel.TransitionState {
		private final int pos;
		private final TransitionStateType type;

		private final int lmCharIndex;
		
		public CharacterNgramTransitionState(int pos, int character, TransitionStateType type) {
			this.pos = pos;
			this.type = type;
			
			if (pos == -1 || character == -1 || type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN || type == TransitionStateType.RMRGN || type == TransitionStateType.RMRGN_HPHN) {
				this.lmCharIndex = spaceCharIndex;
			} else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				this.lmCharIndex = hyphenCharIndex;
			} else {
				this.lmCharIndex = character;
			}
		}
		
		public boolean equals(Object other) {
		    if (other instanceof CharacterNgramTransitionState) {
		    	CharacterNgramTransitionState that = (CharacterNgramTransitionState) other;
		    	if (this.type != that.type) {
		    		return false;
		    	} else if (this.pos != that.pos) {
		    		return false;
		    	} else if (this.lmCharIndex != that.lmCharIndex) {
		    		return false;
		    	} else {
		    		return true;
		    	}
		    } else {
		    	return false;
		    }
		}
		
		public int hashCode() {
			return 1013 * Integer.hashCode(pos) + 1009 * this.type.ordinal() + 1019 * Integer.hashCode(lmCharIndex);
		}
		
		public Collection<Tuple2<TransitionState,Double>> nextLineStartStates() {
			List<Tuple2<TransitionState,Double>> result = new ArrayList<Tuple2<TransitionState,Double>>();
			TransitionStateType type = getType();
			int pos = getPosition();
			
			if (type == TransitionStateType.TMPL) {
				double scoreWithSpace =  Math.log(lm.getCharNgramProb(pos+1, spaceCharIndex));
				if (scoreWithSpace != Double.NEGATIVE_INFINITY) {
					{
						double score = Math.log(LINE_MRGN_PROB) + scoreWithSpace;
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, -1, TransitionStateType.LMRGN), score));
						}
					}
					for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
						if (c != spaceCharIndex && !isPunc[c]) {
							double score = Math.log((1.0 - LINE_MRGN_PROB)) + scoreWithSpace + Math.log(lm.getInsertProb(-1)) + Math.log(lm.getKeepProb(lm.getCharAtPos(pos+2))) + Math.log(lm.getCharNgramProb(pos+2, c));
							if (score != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+2, c, TransitionStateType.TMPL), score));
							}
							
							score = Math.log((1.0 - LINE_MRGN_PROB)) + scoreWithSpace + Math.log(lm.getInsertProb(c));
							if (score != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, c, TransitionStateType.TMPL), score));
							}
						}
					}
					
					int curPos = pos+2;
					double score = Math.log((1.0 - LINE_MRGN_PROB)) + scoreWithSpace;
					
					while(lm.getCharAtPos(curPos) != spaceCharIndex) {
						score += Math.log(lm.getDeleteProb(lm.getCharAtPos(curPos)));
						
						for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
							if (c != spaceCharIndex && !isPunc[c]) {
								double score2 = score + Math.log(lm.getKeepProb(lm.getCharAtPos(curPos+1))) + Math.log(lm.getCharNgramProb(curPos+1, c));
								if (score2 != Double.NEGATIVE_INFINITY) {
									result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(curPos+1, c, TransitionStateType.TMPL), score2));
								}
							}
						}						
						
						curPos += 1;
					}
				}
			} else if (type == TransitionStateType.RMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, -1, TransitionStateType.LMRGN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					if (c != spaceCharIndex && !isPunc[c]) {
						double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(-1)) + Math.log(lm.getKeepProb(lm.getCharAtPos(pos+1))) + Math.log(lm.getCharNgramProb(pos+1, c));
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, c, TransitionStateType.TMPL), score));
						}
						
						score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(c));
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, c, TransitionStateType.TMPL), score));
						}
					}
				}
				
				int curPos = pos+1;
				double score = Math.log((1.0 - LINE_MRGN_PROB));
				
				while(lm.getCharAtPos(curPos) != spaceCharIndex) {
					score += Math.log(lm.getDeleteProb(lm.getCharAtPos(curPos)));
					
					for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
						if (c != spaceCharIndex && !isPunc[c]) {
							double score2 = score + Math.log(lm.getKeepProb(lm.getCharAtPos(curPos+1))) + Math.log(lm.getCharNgramProb(curPos+1, c));
							if (score2 != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(curPos+1, c, TransitionStateType.TMPL), score2));
							}							
						}
					}
					curPos += 1;
				}
			} else if (type == TransitionStateType.RMRGN_HPHN || type == TransitionStateType.RMRGN_HPHN_INIT) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, -1, TransitionStateType.LMRGN_HPHN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					if (c != spaceCharIndex && !isPunc[c]) {
						double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(-1)) + Math.log(lm.getKeepProb(lm.getCharAtPos(pos+1))) + Math.log(lm.getCharNgramProb(pos+1, c));
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, c, TransitionStateType.TMPL), score));
						}
						
						score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(c));
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, c, TransitionStateType.TMPL), score));
						}
					}
				}
				int curPos = pos+1;
				double score = Math.log((1.0 - LINE_MRGN_PROB));
				
				while(lm.getCharAtPos(curPos) != spaceCharIndex) {
					score += Math.log(lm.getDeleteProb(lm.getCharAtPos(curPos)));
					
					for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
						if (c != spaceCharIndex && !isPunc[c]) {
							double score2 = score + Math.log(lm.getKeepProb(lm.getCharAtPos(curPos+1))) + Math.log(lm.getCharNgramProb(curPos+1, c));
							if (score2 != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(curPos+1, c, TransitionStateType.TMPL), score2));
							}
						}
					}
					curPos += 1;
				}
			} else if (type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, -1, TransitionStateType.LMRGN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					if (c != spaceCharIndex && !isPunc[c]) {
						double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(-1)) + Math.log(lm.getKeepProb(lm.getCharAtPos(pos+1))) + Math.log(lm.getCharNgramProb(pos+1, c));
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, c, TransitionStateType.TMPL), score));
						}
						
						score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(c));
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, c, TransitionStateType.TMPL), score));
						}
					}
				}
				int curPos = pos+1;
				double score = Math.log((1.0 - LINE_MRGN_PROB));
				
				while(lm.getCharAtPos(curPos) != spaceCharIndex) {
					score += Math.log(lm.getDeleteProb(lm.getCharAtPos(curPos)));
					
					for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
						if (c != spaceCharIndex && !isPunc[c]) {
							double score2 = score + Math.log(lm.getKeepProb(lm.getCharAtPos(curPos+1))) + Math.log(lm.getCharNgramProb(curPos+1, c));
							if (score2 != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(curPos+1, c, TransitionStateType.TMPL), score2));
							}
						}
					}						
					
					curPos += 1;
				}
			}
			return result;
		}
		
		public double endLogProb() {
			return 0.0;
		}
		
		public Collection<Tuple2<TransitionState,Double>> forwardTransitions() {
			int pos = getPosition();
			TransitionStateType type = getType();
			List<Tuple2<TransitionState,Double>> result = new ArrayList<Tuple2<TransitionState,Double>>();
			if (type == TransitionStateType.LMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, -1, TransitionStateType.LMRGN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					if (c != spaceCharIndex && !isPunc[c]) {
						double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(-1)) + Math.log(lm.getKeepProb(lm.getCharAtPos(pos+1))) + Math.log(lm.getCharNgramProb(pos+1, c));
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, c, TransitionStateType.TMPL), score));
						}
						
						score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(c));
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, c, TransitionStateType.TMPL), score));
						}
					}
				}
				int curPos = pos+1;
				double score = Math.log((1.0 - LINE_MRGN_PROB));
				
				while(lm.getCharAtPos(curPos) != spaceCharIndex) {
					score += Math.log(lm.getDeleteProb(lm.getCharAtPos(curPos)));
					
					for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
						if (c != spaceCharIndex && !isPunc[c]) {
							double score2 = score + Math.log(lm.getKeepProb(lm.getCharAtPos(curPos+1))) + Math.log(lm.getCharNgramProb(curPos+1, c));
							if (score2 != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(curPos+1, c, TransitionStateType.TMPL), score2));
							}
						}
					}						
					
					curPos += 1;
				}
			} else if (type == TransitionStateType.LMRGN_HPHN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, -1, TransitionStateType.LMRGN_HPHN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					if (c != spaceCharIndex && !isPunc[c]) {
						double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(-1)) + Math.log(lm.getKeepProb(lm.getCharAtPos(pos+1))) + Math.log(lm.getCharNgramProb(pos+1, c));
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, c, TransitionStateType.TMPL), score));
						}
						
						score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(c));
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, c, TransitionStateType.TMPL), score));
						}						
					}
				}
				int curPos = pos+1;
				double score = Math.log((1.0 - LINE_MRGN_PROB));
				
				while(lm.getCharAtPos(curPos) != spaceCharIndex) {
					score += Math.log(lm.getDeleteProb(lm.getCharAtPos(curPos)));
					
					for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
						if (c != spaceCharIndex && !isPunc[c]) {
							double score2 = score + Math.log(lm.getKeepProb(lm.getCharAtPos(curPos+1))) + Math.log(lm.getCharNgramProb(curPos+1, c));
							if (score2 != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(curPos+1, c, TransitionStateType.TMPL), score2));
							}
						}
					}						
					
					curPos += 1;
				}
			} else if (type == TransitionStateType.RMRGN) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, -1, TransitionStateType.RMRGN), score));
				}
			} else if (type == TransitionStateType.RMRGN_HPHN) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, -1, TransitionStateType.RMRGN_HPHN), score));
				}
			} else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, -1, TransitionStateType.RMRGN_HPHN), score));
				}
			} else if (type == TransitionStateType.TMPL) {
				{
					double score = Math.log(LINE_MRGN_PROB) + Math.log(1.0 - LINE_END_HYPHEN_PROB) + Math.log(lm.getCharNgramProb(pos+1, spaceCharIndex));
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, -1, TransitionStateType.RMRGN), score));
					}
				}
				{
					double score = Math.log(LINE_MRGN_PROB) + Math.log(LINE_END_HYPHEN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, -2, TransitionStateType.RMRGN_HPHN_INIT), score));
					}
				}
				for (int nextC=0; nextC<lm.getCharacterIndexer().size(); ++nextC) {
					double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(-1)) + Math.log(lm.getKeepProb(lm.getCharAtPos(pos+1))) + Math.log(lm.getCharNgramProb(pos+1, nextC));
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, nextC, TransitionStateType.TMPL), score));
					}
					
					score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getInsertProb(nextC));
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, nextC, TransitionStateType.TMPL), score));
					}
				}
				int curPos = pos+1;
				double score = Math.log((1.0 - LINE_MRGN_PROB));
				
				while(lm.getCharAtPos(curPos) != spaceCharIndex) {
					score += Math.log(lm.getDeleteProb(lm.getCharAtPos(curPos)));
					
					for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
							double score2 = score + Math.log(lm.getKeepProb(lm.getCharAtPos(curPos+1))) + Math.log(lm.getCharNgramProb(curPos+1, c));
							if (score2 != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(curPos+1, c, TransitionStateType.TMPL), score2));
							}						
					}		
					curPos += 1;
				}
			}
			return result;
		}
		
		public int getLmCharIndex() {
			return lmCharIndex;
		}
		
		public GlyphChar getGlyphChar() {
			// Always render the character proposed by the language model
			return new GlyphChar(lmCharIndex, GlyphType.NORMAL_CHAR);
		}
		
		public int getOffset() {
//			throw new Error("Method not implemented");
			return pos;
		}
		
		public int getExposure() {
			throw new Error("Method not implemented");
		}
		
		public int getPosition() {
			return pos;
		}
		
		public TransitionStateType getType() {
			return type;
		}
		
		public int getLanguageIndex() {
			return -1;
		}
	}
	
	public static final double LINE_MRGN_PROB = 0.5;
	public static final double LINE_END_HYPHEN_PROB = 1e-8;
	
	private FixedLanguageModel lm;
	private int spaceCharIndex;
	private int hyphenCharIndex;
	private boolean[] isPunc;

	public FixedAlignTransition(FixedLanguageModel lm) {
		this.lm = lm;
		this.spaceCharIndex = lm.getCharacterIndexer().getIndex(Charset.SPACE);
		this.hyphenCharIndex = lm.getCharacterIndexer().getIndex(Charset.HYPHEN);
		this.isPunc = new boolean[lm.getCharacterIndexer().size()];
		Arrays.fill(this.isPunc, false);
		for (String c : lm.getCharacterIndexer().getObjects()) {
			if(Charset.isPunctuationChar(c))
				isPunc[lm.getCharacterIndexer().getIndex(c)] = true;
		}
	}

	public Collection<Tuple2<TransitionState,Double>> startStates() {
		List<Tuple2<TransitionState,Double>> result = new ArrayList<Tuple2<TransitionState,Double>>();
		result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(-1, -1, TransitionStateType.LMRGN), Math.log(LINE_MRGN_PROB)));
		for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
			if (c != spaceCharIndex && !isPunc[c]) {
			result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(0, c, TransitionStateType.TMPL), Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(0, c))));
			}
		}
		return result;
	}
	
}
