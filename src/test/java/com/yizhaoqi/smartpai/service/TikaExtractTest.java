package com.yizhaoqi.smartpai.service;

import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.sax.BodyContentHandler;
import org.xml.sax.SAXException;

import java.io.*;

/**
 * 临时测试：直接用Tika解析WFANet论文，看原始提取效果
 */
public class TikaExtractTest {
    public static void main(String[] args) throws Exception {
        String pdfPath = "papers/AAAI25-WFANet-用于全色锐化的小波辅助多频注意力网络.pdf";
        String outputPath = "target/wfanet_tika_output.txt";

        File pdfFile = new File(pdfPath);
        if (!pdfFile.exists()) {
            System.err.println("PDF not found at: " + pdfFile.getAbsolutePath());
            System.exit(1);
        }

        System.out.println("Parsing: " + pdfFile.getAbsolutePath());
        System.out.println("File size: " + pdfFile.length() / 1024 / 1024 + "MB");

        long start = System.currentTimeMillis();

        try (InputStream input = new BufferedInputStream(new FileInputStream(pdfFile))) {
            AutoDetectParser parser = new AutoDetectParser();
            BodyContentHandler handler = new BodyContentHandler(-1);
            Metadata metadata = new Metadata();
            ParseContext context = new ParseContext();

            parser.parse(input, handler, metadata, context);

            String text = handler.toString();
            long elapsed = System.currentTimeMillis() - start;

            System.out.println("\n=== METADATA ===");
            for (String name : metadata.names()) {
                System.out.println(name + ": " + metadata.get(name));
            }

            System.out.println("\n=== EXTRACTION RESULT ===");
            System.out.println("Total chars extracted: " + text.length());
            System.out.println("Extraction time: " + elapsed + "ms");

            try (Writer writer = new OutputStreamWriter(new FileOutputStream(outputPath), "UTF-8")) {
                writer.write(text);
            }
            System.out.println("Full output saved to: " + outputPath);

            System.out.println("\n=== FIRST 3000 CHARS ===");
            System.out.println(text.substring(0, Math.min(3000, text.length())));

            int sampleStart = Math.min(5000, text.length());
            System.out.println("\n\n=== CHARS 5000-7000 (paper body section) ===");
            System.out.println(text.substring(sampleStart, Math.min(sampleStart + 2000, text.length())));

        } catch (Exception e) {
            System.err.println("Parse failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
