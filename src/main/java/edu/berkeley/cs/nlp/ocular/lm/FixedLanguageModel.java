package edu.berkeley.cs.nlp.ocular.lm;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.textreader.CharIndexer;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import tberg.murphy.indexer.Indexer;

/**
 * @author Shruti Rijhwani
 */

public class FixedLanguageModel {
	
	private Indexer<String> charIndexer;
	private int maxOrder;
	List<Integer> fixedText;
	
	private int[] counts;
	
	private double[][] substituteProb;
	private double[] insertProb;
	private double[] deleteProb;
	
	private double noInsert;	
	
	public FixedLanguageModel(String fileName) {
		maxOrder = 1;
		charIndexer = new CharIndexer();		
		fixedText = new ArrayList<Integer>();
		
		double eps = 1e-15;		
		double fixedProb = 1-eps;
		noInsert = 1-eps;
		
		charIndexer.getIndex(Charset.HYPHEN);
		int longsIndex = charIndexer.getIndex(Charset.LONG_S);
		
		try {
	      BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
	      
	      while (in.ready()) {
	        String line = in.readLine();
	        for (int i = 0; i < line.length(); i++) {
	        	String toAdd = Character.toString(line.charAt(i));
	        	
	        	int index = charIndexer.getIndex(toAdd);
	        	fixedText.add(index);
	        }
	        fixedText.add(charIndexer.getIndex(Charset.SPACE));
	      }
	      in.close();
	    } catch (IOException e) {
	      throw new RuntimeException(e);
	    }

		charIndexer.getIndex(Charset.SPACE);
		
		int sCharIndex = charIndexer.contains("s") ? charIndexer.getIndex("s") : -1;
		
		double subProb = (1-fixedProb)/(charIndexer.size()-1);
		System.out.println(subProb);
		
		this.substituteProb = new double[charIndexer.size()][charIndexer.size()];
		
		for (int i = 0; i<this.substituteProb.length; i++) {
			Arrays.fill(this.substituteProb[i], subProb);
			this.substituteProb[i][i] = fixedProb;
			
			if (i == sCharIndex) {
				substituteProb[i][i] = fixedProb*3/4;
				substituteProb[i][longsIndex] += fixedProb/4;
			}
		}
		
		this.counts = new int[charIndexer.size()];
		Arrays.fill(counts, 0);
		
		for (int c : fixedText) {
			counts[c] += 1;
		}
		
		this.insertProb = new double[charIndexer.size()];
		double insProb = (1-noInsert)/charIndexer.size();
		Arrays.fill(insertProb, insProb);
		
		double deleteInit = 1e-5;
		this.deleteProb = new double[charIndexer.size()];
		Arrays.fill(deleteProb, deleteInit);
	}
	

	public double getCharNgramProb(int pos, int c) {		
		return substituteProb[this.getCharAtPos(pos)][c];
	}
	
	public double getInsertProb(int c) {
		if (c==-1) {
			return noInsert;
		}
		return insertProb[c];
	}
	
	public double getDeleteProb(int c) {
		return deleteProb[c];	
	}
	
	public double getKeepProb(int c) {
		return 1-deleteProb[c];
	}
	
	public int getCharAtPos(int pos) {
		if (pos < fixedText.size())
			return fixedText.get(pos);
		return charIndexer.getIndex(Charset.SPACE);
	}
	
	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}
	
	public Set<Integer> getActiveCharacters() {
		// TODO Auto-generated method stub
		return null;
	}
	

	public int getMaxOrder() {
		return maxOrder;
	}
	

	public int[] shrinkContext(int[] originalContext) {
		// TODO Auto-generated method stub
		return null;
	}
	

	public boolean containsContext(int[] context) {
		// TODO Auto-generated method stub
		return false;
	}
	
	public void updateProbs(TransitionState[][] decodeStates) {
		
		for (int i = 0; i<this.substituteProb.length; i++) {
			Arrays.fill(this.substituteProb[i], 0.0);
		}		
		Arrays.fill(insertProb, 0.0);
		Arrays.fill(deleteProb, 0.0);
		
		double numInserts = 0;
		int numChars = 0;
		
		int prevPos = -1;
							
		for(TransitionState[] line : decodeStates) {
			for(TransitionState state : line) {
				if (state.getType() != TransitionStateType.TMPL) {
					continue;
				}
				
				numChars += 1;
				
				int curPos = state.getOffset();				
				if (curPos == prevPos+1) {
					substituteProb[this.getCharAtPos(curPos)][state.getLmCharIndex()] += 1;
					prevPos = curPos;
				}
				else if (curPos > prevPos+1) {
					prevPos = prevPos+1;
					while (prevPos < curPos) {
						deleteProb[this.getCharAtPos(prevPos)] += 1;
						prevPos += 1;
					}
					substituteProb[this.getCharAtPos(curPos)][state.getLmCharIndex()] += 1;
					prevPos = curPos;
				}
				else {
					numInserts += 1;
					insertProb[state.getLmCharIndex()] += 1;
				}				
			}
			if (this.getCharAtPos(prevPos+1) == this.getCharacterIndexer().getIndex(Charset.SPACE)) {
				numChars += 1;
				prevPos += 1;
				substituteProb[this.getCharacterIndexer().getIndex(Charset.SPACE)][this.getCharacterIndexer().getIndex(Charset.SPACE)] += 1;
			}
		}
		
		for (int i = 0; i<this.substituteProb.length; i++) {
			for (int j = 0; j<this.substituteProb[i].length; j++) {
				substituteProb[i][j] /= counts[i];
			}
		}
		
		double sum = 0.0;
		this.noInsert = 1-(numInserts/numChars);
		
		sum += noInsert;
		
		for (int i = 0; i<this.insertProb.length; i++) {
			insertProb[i] /= numChars;
			deleteProb[i] /= counts[i];
			sum += insertProb[i];
		}
		
		System.out.println(sum);
	}
	
}