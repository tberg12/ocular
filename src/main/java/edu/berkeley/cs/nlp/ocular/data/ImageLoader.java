package edu.berkeley.cs.nlp.ocular.data;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;

import java.util.List;

public interface ImageLoader {
	
	public static interface Document {
		public String baseName();
		public PixelType[][][] loadLineImages();
		public String[][] loadLineText();
		public boolean useLongS();
	}

	public List<Document> readDataset();
  
}
