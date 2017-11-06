package edu.berkeley.cs.nlp.ocular.lm;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.textreader.CharIndexer;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.lm.SearchStateModel.SearchState;
import edu.berkeley.cs.nlp.ocular.main.LMTrainMain;
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
	
	private int spaceIndex;	

	private static String lmDir = "/Users/shruti/Documents/HistoricalOcr/ocular/lm";
	private static String lmBaseName = "sp";
	
	LanguageModel lm;
	HashMap<Integer, Integer> lmMapping;
	
	private void langModelInit() {

		double eps = 1e-15;		
		double fixedProb = 1-eps;
		noInsert = 1-eps;
		double subProb = eps/(charIndexer.size()-1);	

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
		
		this.lm = LMTrainMain.readLM(lmDir+"/"+lmBaseName+".lmser");
		
		lmMapping = new HashMap<Integer, Integer>();
		
		for (int c=0; c<charIndexer.size(); c++) {
			lmMapping.put(c, this.lm.getCharacterIndexer().getIndex(this.getCharacterIndexer().getObject(c)));
		}
				
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
	
	private String updateModern(List<Transcription> docs) {
		SearchStateModel sm = new SearchStateModel(substituteProb, insertProb, deleteProb, noInsert, this.getCharacterIndexer().size(), docs, lm, lmMapping);
		
		BeamSearch beamSearch = new BeamSearch(5, 3, sm, docs.get(0).getDocLength()+20);
		
		List<SearchState> bestPath = beamSearch.startBeam();
		
		ListIterator<SearchState> li = bestPath.listIterator(bestPath.size());
		
		String text = "";
		
		while (li.hasPrevious()) {
			SearchState searchState = li.previous();
			if (searchState.getContext().length < 1) {
				continue;
			}
			text += this.lm.getCharacterIndexer().getObject(searchState.getContext()[searchState.getContext().length-1]);
		}
		
		return text;		
	}
	
	public void updateProbs(List<TransitionState[][]> nDecodeStates) {
		
		for (int i = 0; i<this.substituteProb.length; i++) {
			Arrays.fill(this.substituteProb[i], 0.1);
		}		
		Arrays.fill(insertProb, 0.1);
		Arrays.fill(deleteProb, 0.1);
		
		double numInserts = 0;
		int numChars = 0;
		
		List<Transcription> docs = new ArrayList<Transcription>();
		List<String> testStrings = new ArrayList<>();
				
		for (TransitionState[][] decodeState : nDecodeStates) {	
			int prevPos = -1;
			
			List<Integer> curDoc = new ArrayList<Integer>();
			
			for(TransitionState[] line : decodeState) {
				for(TransitionState state : line) {
					if (state.getType() != TransitionStateType.TMPL) {
						continue;
					}
					
					curDoc.add(state.getLmCharIndex());
					
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
															
					curDoc.add(this.spaceIndex);					
				}
			}
//			docs.add(new Transcription(curDoc));
			
			String text = "";
			
			for (Integer integer : curDoc) {
				text += this.getCharacterIndexer().getObject(integer);
			}
			
			System.out.println(text);
			testStrings.add(text);
		}	
		
		for (String string : testStrings) {
			List<Integer> cur = new ArrayList<>();
			
			for (int i=0; i < string.length(); i++) {
				cur.add(this.getCharacterIndexer().getIndex(Character.toString(string.charAt(i))));
			}
			
			docs.add(new Transcription(cur));
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
		
		String newFixedText = this.updateModern(docs);
//		String newFixedText = this.testLM();
		this.fixedText = new ArrayList<Integer>();
		
		for (int i = 0; i < newFixedText.length(); i++) {
        	String toAdd = Character.toString(newFixedText.charAt(i));	        	
        	int index = charIndexer.getIndex(toAdd);
        	fixedText.add(index);
        }		
		
		this.printFixedText();
//		this.langModelInit();
	}
	
	public String testLM () {
		List<Transcription> docs = new ArrayList<Transcription>();
		String[] testStrings = {"ltey mato. Todo lo qual te eſlentaiy hace libTe de todo reſpectory obligacion; y aſi, puedes decir de la hiſtoriae todo aquello qe te pareciere, ſin temor que te calumien por el mala ni te premien por el bien que diyeres d ella. Solo qiuſiera dartela mondaa y deſnuda, ſin el hornato de Prologo, ni de la inumerabilidadry catalogo,de los acoſtumbrados sonetos, bpigramaSs y blogios que al principio de los libros ſuele ponerſe. Porque te ſe dqtuequd ", "rey mat. Todo lo cual te esenta y hace libre de todo respecto y obligacion; y aſi, puedes decir de la hiſtorie todo aquello que te pareciere, ſin temor que te calunien por el mal ni te premien por el bien que diyeres d ella. Solo qisiera dartela monda y desnuda, ſin el ornato de prologo, ni de la inumerabilidad y catalogo de los acoſtumbrados sonetos, epigramas y elogios ue al principio de los libros ſuele ponerſe. Porque te ſe que  qu "};
		
		for (String string : testStrings) {
			List<Integer> cur = new ArrayList<>();
			
			for (int i=0; i < string.length(); i++) {
				cur.add(this.getCharacterIndexer().getIndex(Character.toString(string.charAt(i))));
			}
			
			docs.add(new Transcription(cur));
		}
		
		String text = this.updateModern(docs);
		return text;
		
	}
	
}