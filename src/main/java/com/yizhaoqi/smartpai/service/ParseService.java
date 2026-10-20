package com.yizhaoqi.smartpai.service;

import com.yizhaoqi.smartpai.model.DocumentVector;
import com.yizhaoqi.smartpai.repository.DocumentVectorRepository;
import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.sax.BodyContentHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.xml.sax.SAXException;

import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;

@Service
public class ParseService {

    private static final Logger logger = LoggerFactory.getLogger(ParseService.class);

    @Autowired
    private DocumentVectorRepository documentVectorRepository;

    @Value("${file.parsing.chunk-size}")
    private int chunkSize;

    @Value("${file.parsing.parent-chunk-size:1048576}")
    private int parentChunkSize;

    @Value("${file.parsing.buffer-size:8192}")
    private int bufferSize;

    @Value("${file.parsing.max-memory-threshold:0.8}")
    private double maxMemoryThreshold;

    public ParseService() {
        // 无需初始化，StandardTokenizer是静态方法
    }

    /**
     * 以流式方式解析文件，将内容分块并保存到数据库，以避免OOM。
     * 采用"父文档-子切片"策略，按论文章节语义分块。
     *
     * @param fileMd5    文件的MD5哈希值，用于唯一标识文件
     * @param fileStream 文件输入流，用于读取文件内容
     * @param userId     上传用户ID
     * @param orgTag     组织标签
     * @param isPublic   是否公开
     * @param fileName   原始文件名（用于提取文档标题）
     * @throws IOException   如果文件读取过程中发生错误
     * @throws TikaException 如果文件解析过程中发生错误
     */
    public void parseAndSave(String fileMd5, InputStream fileStream,
            String userId, String orgTag, boolean isPublic, String fileName) throws IOException, TikaException {
        logger.info("开始流式解析文件，fileMd5: {}, userId: {}, orgTag: {}, isPublic: {}, fileName: {}",
                fileMd5, userId, orgTag, isPublic, fileName);

        checkMemoryThreshold();

        try (BufferedInputStream bufferedStream = new BufferedInputStream(fileStream, bufferSize)) {
            StreamingContentHandler handler = new StreamingContentHandler(fileMd5, userId, orgTag, isPublic, fileName);
            Metadata metadata = new Metadata();
            ParseContext context = new ParseContext();
            AutoDetectParser parser = new AutoDetectParser();

            parser.parse(bufferedStream, handler, metadata, context);

            logger.info("文件流式解析和入库完成，fileMd5: {}", fileMd5);

        } catch (SAXException e) {
            logger.error("文档解析失败，fileMd5: {}", fileMd5, e);
            throw new RuntimeException("文档解析失败", e);
        }
    }

    public void parseAndSave(String fileMd5, InputStream fileStream,
            String userId, String orgTag, boolean isPublic) throws IOException, TikaException {
        parseAndSave(fileMd5, fileStream, userId, orgTag, isPublic, null);
    }

    public void parseAndSave(String fileMd5, InputStream fileStream) throws IOException, TikaException {
        parseAndSave(fileMd5, fileStream, "unknown", "DEFAULT", false, null);
    }

    private void checkMemoryThreshold() {
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;

        double memoryUsage = (double) usedMemory / maxMemory;

        if (memoryUsage > maxMemoryThreshold) {
            logger.warn("内存使用率过高: {:.2f}%, 触发垃圾回收", memoryUsage * 100);
            System.gc();

            usedMemory = runtime.totalMemory() - runtime.freeMemory();
            memoryUsage = (double) usedMemory / maxMemory;

            if (memoryUsage > maxMemoryThreshold) {
                throw new RuntimeException("内存不足，无法处理大文件。当前使用率: " +
                    String.format("%.2f%%", memoryUsage * 100));
            }
        }
    }

    // ======================== 章节检测（内容级） ========================

    private static class SectionWithContent {
        private final String title;
        private final int level;
        private final String content;
        SectionWithContent(String title, int level, String content) {
            this.title = title;
            this.level = level;
            this.content = content;
        }
        String getTitle() { return title; }
        int getLevel() { return level; }
        String getContent() { return content; }
    }

    /**
     * Tika PDF 输出无干净换行且常将单词粘连（如 "IEEEAbstract—"），
     * 因此所有模式均不依赖前置 \\b 词边界，改用后置 \\b 或具体上下文来防过匹配。
     */
    private static final Pattern[] SECTION_BOUNDARIES = {
        // 0) 章节标题与正文无分隔符：AAAI 论文提取结果 "AbstractP"、"IntroductionH"、"MethodologyI"
        //    Tika 将标题和首句粘合，无空格/冒号/词边界，靠后跟大写字母识别
        Pattern.compile("(Abstract|Introduction|Methodology|Experiments|Conclusion|"
                + "Acknowledgments?)(?=[A-Z])"),
        Pattern.compile("(Related Work|Proposed Method|Proposed Framework|"
                + "Experimental Results|Results and Discussion)(?=[A-Z])"),
        // 1) "Abstract—" 或 "Abstract:" — IEEE 论文, Abstract 常与前文粘连
        Pattern.compile("(Abstract)[—–\\-:]"),
        // 2) 罗马数字 + 已知章节名: "I.INTRODUCTION"(无空格) 或 "II. RELATED WORK"
        //    优先于全大写模式，避免 "IV. EXPERIMENTS" 被 "EXPERIMENTS" 覆盖
        Pattern.compile("((?:I{1,3}|IV|V|VI{0,3}|IX|X)\\.\\s*(?:INTRODUCTION|RELATED WORK|"
                + "PROPOSED METHOD|PROPOSED FRAMEWORK|METHODOLOGY|"
                + "EXPERIMENTAL RESULTS|EXPERIMENTS|RESULTS AND DISCUSSION|"
                + "RESULTS|DISCUSSION|CONCLUSION|REFERENCES|ACKNOWLEDGMENT|"
                + "APPENDIX|BIBLIOGRAPHY))"),
        // 3) 全大写知名章节名（去掉前置 \\b，保留后置 \\b 防部分匹配）
        Pattern.compile("(INTRODUCTION|METHODOLOGY|EXPERIMENTS|CONCLUSION|REFERENCES|"
                + "ACKNOWLEDGMENT|APPENDIX|BIBLIOGRAPHY|DISCUSSION|"
                + "RELATED WORK|PROPOSED METHOD|PROPOSED FRAMEWORK|"
                + "EXPERIMENTAL RESULTS|RESULTS AND DISCUSSION)\\b"),
        // 4) 数字编号章节: "1. Introduction", "2. Related Work", "3. Proposed Method" 等
        Pattern.compile("(\\d+)\\.\\s+(Introduction|Related Work|Proposed Method|Proposed Framework|"
                + "Methodology|Experimental Results|Experiments|Results and Discussion|"
                + "Conclusion|Preliminaries|Background|Approach|Ablation Study|Limitations|"
                + "Problem Definition|Problem Formulation|Implementation Details)"),
    };

    // 已知章节名列表，用于从匹配中提取干净的标题
    private static final String[] KNOWN_SECTIONS = {
        "INTRODUCTION", "RELATED WORK", "PROPOSED METHOD", "PROPOSED FRAMEWORK",
        "METHODOLOGY", "EXPERIMENTAL RESULTS", "EXPERIMENTS", "RESULTS AND DISCUSSION",
        "RESULTS", "DISCUSSION", "CONCLUSION", "REFERENCES", "ACKNOWLEDGMENT",
        "APPENDIX", "BIBLIOGRAPHY"
    };

    private String extractDocTitle(String fileName) {
        if (fileName == null || fileName.isBlank()) return "unknown";
        int dotIdx = fileName.lastIndexOf('.');
        return dotIdx > 0 ? fileName.substring(0, dotIdx) : fileName;
    }

    /**
     * 在文本流中搜索章节边界并按章节分割。
     * 因为 Tika PDF 提取的输出常常没有换行，所以用 Matcher.find()
     * 在全文搜索已知模式，在匹配位置切分。
     */
    private List<SectionWithContent> splitIntoSections(String text, String defaultTitle, int defaultLevel) {
        List<SectionWithContent> sections = new ArrayList<>();
        if (text == null || text.isBlank()) return sections;

        // 收集所有章节边界位置
        List<SplitPoint> splits = new ArrayList<>();

        for (Pattern pattern : SECTION_BOUNDARIES) {
            Matcher m = pattern.matcher(text);
            while (m.find()) {
                int pos = m.start();
                // 去重：同一位置或附近（≤15 字符）只保留第一个匹配
                // 场景: "IV. EXPERIMENTS"(模式3) 与 "EXPERIMENTS"(模式2) 相差仅 4 字符
                boolean dup = false;
                for (SplitPoint sp : splits) {
                    if (Math.abs(sp.pos - pos) < 15) { dup = true; break; }
                }
                if (!dup) {
                    String title = extractTitleFromMatch(m);
                    splits.add(new SplitPoint(pos, title, 1));
                }
            }
        }

        // 按位置排序
        splits.sort(Comparator.comparingInt(sp -> sp.pos));

        // 按边界切分
        int cursor = 0;
        String currentTitle = defaultTitle;
        int currentLevel = defaultLevel;

        for (SplitPoint sp : splits) {
            if (sp.pos <= cursor) {
                // 用新标题覆盖（同一位置的更精确标题）
                currentTitle = sp.title;
                currentLevel = sp.level;
                continue;
            }
            // 保存 cursor → sp.pos 的内容
            if (sp.pos > cursor) {
                String content = text.substring(cursor, sp.pos).trim();
                if (!content.isEmpty()) {
                    sections.add(new SectionWithContent(currentTitle, currentLevel, content));
                }
            }
            currentTitle = sp.title;
            currentLevel = sp.level;
            cursor = sp.pos;
        }

        // 剩余内容
        if (cursor < text.length()) {
            String content = text.substring(cursor).trim();
            if (!content.isEmpty()) {
                sections.add(new SectionWithContent(currentTitle, currentLevel, content));
            }
        }

        return sections;
    }

    private String extractTitleFromMatch(Matcher m) {
        String raw = m.group().trim();
        // "Abstract—" → "Abstract"
        if (raw.startsWith("Abstract")) return "Abstract";
        // 全大写标题直接返回
        if (raw.matches("^[A-Z]+$")) return raw;
        // 罗马数字: "I.INTRODUCTIONH" → "I. INTRODUCTION"
        if (raw.matches("^[IVXLCDM]+\\.\\s*[A-Z].*")) {
            String[] parts = raw.split("\\.\\s*", 2);
            if (parts.length == 2) {
                String after = parts[1].trim();
                // 从已知章节名列表中匹配
                for (String known : KNOWN_SECTIONS) {
                    if (after.startsWith(known)) {
                        return parts[0] + ". " + known;
                    }
                }
                // 备选：取首段大写字母序列（可能带噪声如 INTRODUCTIONH）
                return parts[0] + ". " + after.replaceAll("([A-Z]+).*", "$1");
            }
        }
        return raw.replaceAll("\\s+", " ").trim();
    }

    private static class SplitPoint {
        final int pos;
        final String title;
        final int level;
        SplitPoint(int pos, String title, int level) {
            this.pos = pos;
            this.title = title;
            this.level = level;
        }
    }

    // ======================== 流式处理器 ========================

    private class StreamingContentHandler extends BodyContentHandler {
        private final StringBuilder buffer = new StringBuilder();
        private final String fileMd5;
        private final String userId;
        private final String orgTag;
        private final boolean isPublic;
        private final String docTitle;
        private int savedChunkCount = 0;
        private String currentSectionTitle = "";
        private int currentSectionLevel = 0;

        public StreamingContentHandler(String fileMd5, String userId, String orgTag, boolean isPublic, String fileName) {
            super(-1);
            this.fileMd5 = fileMd5;
            this.userId = userId;
            this.orgTag = orgTag;
            this.isPublic = isPublic;
            this.docTitle = extractDocTitle(fileName);
        }

        @Override
        public void characters(char[] ch, int start, int length) {
            buffer.append(ch, start, length);
            if (buffer.length() >= parentChunkSize) {
                processParentChunk();
            }
        }

        @Override
        public void endDocument() {
            if (buffer.length() > 0) {
                processParentChunk();
            }
        }

        private void processParentChunk() {
            String parentChunkText = buffer.toString();

            // 按章节分割父块，当前章节信息跨块传递
            List<SectionWithContent> sections = splitIntoSections(parentChunkText, currentSectionTitle, currentSectionLevel);

            for (SectionWithContent section : sections) {
                List<String> childChunks = ParseService.this.splitTextIntoChunksWithSemantics(section.getContent(), chunkSize);
                savedChunkCount = ParseService.this.saveChildChunks(fileMd5, childChunks, userId, orgTag, isPublic,
                        savedChunkCount, docTitle, section.getTitle(), section.getLevel());

                // 更新当前章节信息，供下一个父块继承
                currentSectionTitle = section.getTitle();
                currentSectionLevel = section.getLevel();
            }

            buffer.setLength(0);
        }
    }

    // ======================== 子切片保存 ========================

    private int saveChildChunks(String fileMd5, List<String> chunks,
            String userId, String orgTag, boolean isPublic, int startingChunkId,
            String docTitle, String sectionTitle, Integer sectionLevel) {
        int currentChunkId = startingChunkId;
        for (String chunk : chunks) {
            currentChunkId++;
            var vector = new DocumentVector();
            vector.setFileMd5(fileMd5);
            vector.setChunkId(currentChunkId);
            vector.setTextContent(chunk);
            vector.setUserId(userId);
            vector.setOrgTag(orgTag);
            vector.setPublic(isPublic);
            vector.setDocTitle(docTitle);
            vector.setSectionTitle(sectionTitle);
            vector.setSectionLevel(sectionLevel);
            documentVectorRepository.save(vector);
        }
        logger.debug("成功保存 {} 个子切片到数据库，当前章节: {}", chunks.size(), sectionTitle);
        return currentChunkId;
    }

    // ======================== 语义分块（原有逻辑不变） ========================

    private List<String> splitTextIntoChunksWithSemantics(String text, int chunkSize) {
        List<String> chunks = new ArrayList<>();
        String[] paragraphs = text.split("\n\n+");
        StringBuilder currentChunk = new StringBuilder();

        for (String paragraph : paragraphs) {
            if (paragraph.length() > chunkSize) {
                if (currentChunk.length() > 0) {
                    chunks.add(currentChunk.toString().trim());
                    currentChunk = new StringBuilder();
                }
                List<String> sentenceChunks = splitLongParagraph(paragraph, chunkSize);
                chunks.addAll(sentenceChunks);
            }
            else if (currentChunk.length() + paragraph.length() > chunkSize) {
                if (currentChunk.length() > 0) {
                    chunks.add(currentChunk.toString().trim());
                }
                currentChunk = new StringBuilder(paragraph);
            }
            else {
                if (currentChunk.length() > 0) {
                    currentChunk.append("\n\n");
                }
                currentChunk.append(paragraph);
            }
        }

        if (currentChunk.length() > 0) {
            chunks.add(currentChunk.toString().trim());
        }

        return chunks;
    }

    private List<String> splitLongParagraph(String paragraph, int chunkSize) {
        List<String> chunks = new ArrayList<>();
        String[] sentences = paragraph.split("(?<=[。！？；])|(?<=[.!?;])\\s+");
        StringBuilder currentChunk = new StringBuilder();

        for (String sentence : sentences) {
            if (currentChunk.length() + sentence.length() > chunkSize) {
                if (currentChunk.length() > 0) {
                    chunks.add(currentChunk.toString().trim());
                    currentChunk = new StringBuilder();
                }
                if (sentence.length() > chunkSize) {
                    chunks.addAll(splitLongSentence(sentence, chunkSize));
                } else {
                    currentChunk.append(sentence);
                }
            } else {
                currentChunk.append(sentence);
            }
        }

        if (currentChunk.length() > 0) {
            chunks.add(currentChunk.toString().trim());
        }

        return chunks;
    }

    private List<String> splitLongSentence(String sentence, int chunkSize) {
        List<String> chunks = new ArrayList<>();

        try {
            List<Term> termList = StandardTokenizer.segment(sentence);
            StringBuilder currentChunk = new StringBuilder();
            for (Term term : termList) {
                String word = term.word;
                if (currentChunk.length() + word.length() > chunkSize && !currentChunk.isEmpty()) {
                    chunks.add(currentChunk.toString());
                    currentChunk = new StringBuilder();
                }
                currentChunk.append(word);
            }
            if (!currentChunk.isEmpty()) {
                chunks.add(currentChunk.toString());
            }
            logger.debug("HanLP智能分词成功，原文长度: {}, 分词数: {}, 分块数: {}",
                    sentence.length(), termList.size(), chunks.size());
        } catch (Exception e) {
            logger.warn("HanLP分词异常: {}, 使用字符分割作为备用方案", e.getMessage());
            chunks = splitByCharacters(sentence, chunkSize);
        }

        return chunks;
    }

    private List<String> splitByCharacters(String sentence, int chunkSize) {
        List<String> chunks = new ArrayList<>();
        StringBuilder currentChunk = new StringBuilder();

        for (int i = 0; i < sentence.length(); i++) {
            char c = sentence.charAt(i);
            if (currentChunk.length() + 1 > chunkSize && !currentChunk.isEmpty()) {
                chunks.add(currentChunk.toString());
                currentChunk = new StringBuilder();
            }
            currentChunk.append(c);
        }

        if (!currentChunk.isEmpty()) {
            chunks.add(currentChunk.toString());
        }

        return chunks;
    }
}
