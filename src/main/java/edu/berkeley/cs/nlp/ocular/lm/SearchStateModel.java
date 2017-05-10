package edu.berkeley.cs.nlp.ocular.lm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.main.LMTrainMain;
import javafx.util.Pair;

public class SearchStateModel {

	public class SearchState {
		private int[] context;
		private int[] positions;
		
		public SearchState(int[] context, int[] positions) {
			this.context = context;
			this.positions = positions;
		}
		
		public double getEndScore() {
			double endScore = 0.0;
			
			for (int i=0; i<positions.length; i++) {
				int curPos = positions[i]+1;
				
				while (curPos < docs.get(i).getDocLength()) {
					endScore += Math.log(insertProb[docs.get(i).getCharAt(curPos)]);
					curPos +=1;
				}
			}			
			return endScore;
		}
		
		public List<Pair<SearchState,Double>> getSuccessors() {
			List<Pair<SearchState,Double>> nextStates = new ArrayList<Pair<SearchState,Double>>();
			
			for (int[] s : switches) {
				
				int[] newPos = new int[positions.length];
				
				boolean check = true;
				
				for (int i=0; i<s.length; i++) {
					if(s[i] == 1) {
						if (positions[i] >= docs.get(i).getDocLength()-1) {
							check = false;
							break;
						}
						newPos[i] = positions[i]+1;
					}
					else {
						newPos[i] = positions[i];
					}
				}
				
				if (!check) {
					continue;
				}

				for (int c=0; c < numChars; c++) {
					SearchState newState = new SearchState(createNewContext(context, lmMapping.get(c)), newPos);					
					double score = 0.0;
					
					for (int i=0; i<s.length; i++) {
						if (s[i]==1) {
							score += Math.log(substituteProb[c][docs.get(i).getCharAt(newPos[i])]);
						}
						else {
							score += Math.log(deleteProb[c]);
						}
					}
					score += Math.log(lm.getCharNgramProb(context, lmMapping.get(c)));		
					
					if (score != Double.NEGATIVE_INFINITY) {
					nextStates.add(new Pair<SearchStateModel.SearchState, Double>(newState, score));
					}
				}
			}			
			
			return nextStates;			
		}
		
		public List<Pair<SearchState, Double>> getEpsilons() {
			List<Pair<SearchState,Double>> nextStates = new ArrayList<Pair<SearchState,Double>>();
			
			for (int[] s : switches) {
				
				int[] newPos = new int[positions.length];
				
				boolean check = true;
				boolean checkForOne = false;
				
				for (int i=0; i<s.length; i++) {
					if(s[i] == 1) {
						checkForOne = true;
						if (positions[i] >= docs.get(i).getDocLength()-1) {
							check = false;
							break;
						}
						newPos[i] = positions[i]+1;
					}
					else {
						newPos[i] = positions[i];
					}
				}
				
				if (!check || !checkForOne) {
					continue;
				}

				SearchState newState = new SearchState(context, newPos);					
				double score = 0.0;
				
				for (int i=0; i<s.length; i++) {
					if (s[i]==1) {
						score += Math.log(insertProb[docs.get(i).getCharAt(newPos[i])]);
					}
					else {
						score += Math.log(noInsert);
					}
				}	
				if (score != Double.NEGATIVE_INFINITY) {				
				nextStates.add(new Pair<SearchStateModel.SearchState, Double>(newState, score));
				}
			}

			return nextStates;
		}
		
		public int[] getContext() {
			return context;
		}
		
		public int hashCode() {
			return 1013 * Arrays.hashCode(context) + 1009 * Arrays.hashCode(positions);
		}
		
		public boolean equals(Object other) {
		    if (other instanceof SearchState) {
		    	SearchState that = (SearchState) other;
		    	if (!Arrays.equals(this.context, that.context)) {
		    		return false;
		    	} else if (!Arrays.equals(this.positions, that.positions)) {
		    		return false;
		    	} else {
		    		return true;
		    	}
		    } else {
		    	return false;
		    }
		}
		
	}
	
	private int ngramOrder = 6;
	private double[][] substituteProb;
	private double[] insertProb;
	private double[] deleteProb;
	private double noInsert;
	private int numChars;
	private int numDocs;
	List<Transcription> docs;
	List<int[]> switches;
	HashMap<Integer, Integer> lmMapping;
	
	LanguageModel lm;
	
	public SearchStateModel(double[][] substituteProb, double[] insertProb, double[] deleteProb, double noInsert, int numChars, List<Transcription> docs, LanguageModel lm, HashMap<Integer, Integer> lmMapping) {
		this.substituteProb = substituteProb;
		this.insertProb = insertProb;
		this.deleteProb = deleteProb;
		this.noInsert = noInsert;
		this.numChars = numChars;
		this.numDocs = docs.size();
		this.docs = docs;
		this.lmMapping = lmMapping;
		
		this.switches = new ArrayList<int[]>();
		
		this.lm = lm;
		
		for (int i=0; i < Math.pow(2, numDocs); i++) {
	        String binString = Integer.toBinaryString(i);
	        int[] sw = new int[numDocs];
	        
	        for (int j=0; j<sw.length; j++) {
	        	if (binString.length() >= (numDocs-j)) {
	        		sw[j] = Integer.parseUnsignedInt(binString.substring(binString.length()-numDocs+j,binString.length()-numDocs+j+1));
	        	}
	        	else {
	        		sw[j] = 0;
	        	}
	        }
	        switches.add(sw);
		}
	}
	
	public SearchState startState() {
		int[] pos = new int[numDocs];
		Arrays.fill(pos, -1);
 		return new SearchState(new int[0], pos);
	}
	
	private int[] createNewContext(int[] context, int c) {
		if (context.length < ngramOrder-1) {
			int[] newContext = new int[context.length+1];
			
			for (int i=0; i<context.length; i++) {
				newContext[i] = context[i];
			}
			newContext[newContext.length-1] = c;
			return newContext;
		}
		int[] newContext = new int[context.length];
		
		for (int i=1; i<context.length; i++) {
			newContext[i-1] = context[i];
		}
		newContext[newContext.length-1] = c;
		return newContext;
	}

}
