package edu.berkeley.cs.nlp.ocular.data;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;

import java.util.List;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public interface ImageLoader {
	
	public static interface Document {
		public String baseName();
		public PixelType[][][] loadLineImages();
		public String[][] loadLineText();
		public List<String> loadLmText();
	}

	public List<Document> readDataset();
  
}
