package edu.berkeley.cs.nlp.ocular.lm;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.textreader.CharIndexer;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.model.transition.FixedAlignTransition;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import javafx.util.Pair;
import tberg.murphy.indexer.Indexer;
import tberg.murphy.math.m;

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
	
	private int spaceIndex;
	
	private void langModelInit() {

		double eps = 1e-15;		
		double fixedProb = 1-eps;
		noInsert = 1-1e-15;
		double subProb = (1-fixedProb)/(charIndexer.size()-1);
		System.out.println(subProb);	

		int sCharIndex = charIndexer.contains("s") ? charIndexer.getIndex("s") : -1;
		int longsIndex = charIndexer.getIndex(Charset.LONG_S);
		
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
	
	
	public FixedLanguageModel(String fileName) {
		maxOrder = 1;
		charIndexer = new CharIndexer();		
		fixedText = new ArrayList<Integer>();
				
		charIndexer.getIndex(Charset.HYPHEN);
		charIndexer.getIndex(Charset.LONG_S);
		
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

		this.spaceIndex = charIndexer.getIndex(Charset.SPACE);
				
		langModelInit();
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
		return this.spaceIndex;
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
	
	private void printFixedText() {
		System.out.println("\n");
		for (int character : this.fixedText) {
			System.out.print(this.charIndexer.getObject(character));
//			System.out.print(character + " ");
		}
	}
	
	private void updateModern(List<List<TransitionState>> nDocStates) {
		List<Integer> newText = new ArrayList<Integer>();
		int docPos[] = new int[nDocStates.size()];
		Arrays.fill(docPos, 0);
		
		for (int pos=0; pos<this.fixedText.size(); pos++) {
			HashMap<String, Pair<Integer, Double>> score = new HashMap<>();
			HashMap<String, Pair<Integer, Double>> insert = new HashMap<>();
			
			for (int doc=0; doc<nDocStates.size(); doc++) {
				List<TransitionState> curDoc = nDocStates.get(doc);
				if (curDoc.size()-1 < docPos[doc]) {
					continue;
				}
				TransitionState state = curDoc.get(docPos[doc]);
				
				if (state.getOffset() > pos) {
					Pair<Integer, Double> curValue = score.get("$");
					
					if (curValue == null) {
						curValue = new Pair<Integer, Double>(0, 0.0);
					}
					
					score.put("$", new Pair<Integer, Double>(curValue.getKey()+1, this.deleteProb[fixedText.get(pos)]));
					continue;
				}
				
				if (state.getOffset() == pos) {
					String charAtPos = Integer.toString(state.getLmCharIndex());
					Pair<Integer, Double> curValue = score.get(charAtPos);
					
					if (curValue == null) {
						curValue = new Pair<Integer, Double>(0, 0.0);
					}
					
					score.put(charAtPos, new Pair<Integer, Double>(curValue.getKey()+1, this.substituteProb[fixedText.get(pos)][state.getLmCharIndex()]));
				}
				
				int curDocPos = docPos[doc] + 1;
				String posInsert = "";
				double insertProb = 1.0;
				
				while (curDocPos < curDoc.size()) {
					state = curDoc.get(curDocPos);
					if (state.getOffset() != pos) {
						break;
					}
					posInsert += Integer.toString(state.getLmCharIndex()) + " ";
					insertProb *= this.insertProb[state.getLmCharIndex()];
					curDocPos += 1;
				}
				
				if (posInsert.length() == 0) {
					posInsert = "$";
					insertProb = this.noInsert;
				}
					
				Pair<Integer, Double> curValue = insert.get(posInsert.trim());
				
				if (curValue == null) {
					curValue = new Pair<Integer, Double>(0, 0.0);
				}
				
				insert.put(posInsert.trim(), new Pair<Integer, Double>(curValue.getKey()+1, insertProb));
				
				docPos[doc] = curDocPos;
			}
			
			double max = 0.0;
			int count = 0;
			String toPut = "";
			
			for (String key : score.keySet()) {
				Pair<Integer, Double> val = score.get(key);
				if (val.getKey() > count) {
					max = val.getValue();
					toPut = key;
					count = val.getKey();
				}
				else if (val.getKey() == count && val.getValue() > max) {
					max = val.getValue();
					toPut = key;
				}
			}
			
			if (toPut != "$" && toPut != "") {
				newText.add(Integer.parseInt(toPut));
			}
			
			max = 0.0;
			toPut = "";
			count = 0;
			
			for (String key : insert.keySet()) {
				Pair<Integer, Double> val = insert.get(key);
				if (val.getKey() > count) {
					max = val.getValue();
					toPut = key;
					count = val.getKey();
				}
				else if (val.getKey() == count && val.getValue() > max) {
					max = val.getValue();
					toPut = key;
				}
			}
			
			String[] spl = toPut.split(" ");
			
			for (String val : spl) {
				if (val == "$" || val == "") {
					continue;
				}
				newText.add(Integer.parseInt(val));
			}
		}
		this.printFixedText();
		this.fixedText = newText;
	}
	
	public void updateProbs(List<TransitionState[][]> nDecodeStates) {
		
		for (int i = 0; i<this.substituteProb.length; i++) {
			Arrays.fill(this.substituteProb[i], 0.0);
		}		
		Arrays.fill(insertProb, 0.0);
		Arrays.fill(deleteProb, 0.0);
		
		double numInserts = 0;
		int numChars = 0;
		
		List<List<TransitionState>> flattened = new ArrayList<List<TransitionState>>();
				
		for (TransitionState[][] decodeState : nDecodeStates) {	
			int prevPos = -1;
			
			List<TransitionState> curDoc = new ArrayList<TransitionState>();
			
			for(TransitionState[] line : decodeState) {
				for(TransitionState state : line) {
					if (state.getType() != TransitionStateType.TMPL) {
						continue;
					}
					
					curDoc.add(state);
					
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
				if (this.getCharAtPos(prevPos+1) == this.spaceIndex) {
					numChars += 1;
					prevPos += 1;
					substituteProb[this.spaceIndex][this.spaceIndex] += 1;
					
					TransitionState spaceState = new TempTransitionState(prevPos, spaceIndex, TransitionStateType.TMPL);											curDoc.add(spaceState);					
				}
			}
			flattened.add(curDoc);
		}		
		
		for (int i = 0; i<this.substituteProb.length; i++) {

			double sum = 0.0;			
			for (int j = 0; j<this.substituteProb[i].length; j++) {
				sum += substituteProb[i][j];
			}			
			if (sum == 0) {
				continue;
			}			
			for (int j = 0; j<this.substituteProb[i].length; j++) {				
				substituteProb[i][j] /= sum;
			}
		}
		
		this.noInsert = 1-(numInserts/numChars);
		
		for (int i = 0; i<this.insertProb.length; i++) {
			insertProb[i] /= numChars;
			if (counts[i] == 0) {
				deleteProb[i] = 0;
				continue;
			}
			deleteProb[i] /= counts[i];
		}
		
		this.updateModern(flattened);
		this.printFixedText();
		this.langModelInit();
	}
	
}