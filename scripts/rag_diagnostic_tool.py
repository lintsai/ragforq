#!/usr/bin/env python3
"""
RAG系統診斷和修復工具
用於檢測和解決常見的RAG問題
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Q_DRIVE_PATH, VECTOR_DB_PATH, SUPPORTED_FILE_TYPES
from indexer.document_indexer import DocumentIndexer
from utils.file_parsers import FileParser

logger = logging.getLogger(__name__)

class RAGDiagnosticTool:
    """RAG系統診斷工具"""
    
    def __init__(self):
        self.issues = []
        self.fixes_applied = []
        
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """運行完整診斷"""
        logger.info("開始RAG系統診斷...")
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "issues": [],
            "recommendations": [],
            "system_status": "unknown"
        }
        
        # 1. 檢查Q槽訪問
        q_drive_status = self._check_q_drive_access()
        results["q_drive_status"] = q_drive_status
        
        # 2. 檢查向量數據庫
        vector_db_status = self._check_vector_database()
        results["vector_db_status"] = vector_db_status
        
        # 3. 檢查文件編碼問題
        encoding_issues = self._check_file_encoding_issues()
        results["encoding_issues"] = encoding_issues
        
        # 4. 檢查索引完整性
        index_integrity = self._check_index_integrity()
        results["index_integrity"] = index_integrity
        
        # 5. 檢查動態RAG文件掃描
        dynamic_scan_status = self._check_dynamic_rag_scan()
        results["dynamic_scan_status"] = dynamic_scan_status
        
        # 6. 性能分析
        performance_analysis = self._analyze_performance()
        results["performance_analysis"] = performance_analysis
        
        # 生成建議
        results["recommendations"] = self._generate_recommendations(results)
        
        # 確定系統狀態
        results["system_status"] = self._determine_system_status(results)
        
        logger.info(f"診斷完成，系統狀態: {results['system_status']}")
        return results
    
    def _check_q_drive_access(self) -> Dict[str, Any]:
        """檢查Q槽訪問狀態"""
        logger.info("檢查Q槽訪問狀態...")
        
        status = {
            "accessible": False,
            "path": Q_DRIVE_PATH,
            "file_count": 0,
            "supported_file_count": 0,
            "issues": []
        }
        
        try:
            if not os.path.exists(Q_DRIVE_PATH):
                status["issues"].append(f"Q槽路徑不存在: {Q_DRIVE_PATH}")
                return status
            
            if not os.access(Q_DRIVE_PATH, os.R_OK):
                status["issues"].append(f"Q槽路徑無讀取權限: {Q_DRIVE_PATH}")
                return status
            
            # 統計文件
            total_files = 0
            supported_files = 0
            
            for root, dirs, files in os.walk(Q_DRIVE_PATH):
                total_files += len(files)
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in SUPPORTED_FILE_TYPES:
                        supported_files += 1
                
                # 限制掃描深度避免過長時間
                if root.replace(Q_DRIVE_PATH, '').count(os.sep) >= 3:
                    dirs[:] = []
            
            status["accessible"] = True
            status["file_count"] = total_files
            status["supported_file_count"] = supported_files
            
            if supported_files == 0:
                status["issues"].append("Q槽中沒有找到支持的文件類型")
            
        except Exception as e:
            status["issues"].append(f"檢查Q槽時出錯: {str(e)}")
        
        return status
    
    def _check_vector_database(self) -> Dict[str, Any]:
        """檢查向量數據庫狀態"""
        logger.info("檢查向量數據庫狀態...")
        
        status = {
            "exists": False,
            "path": VECTOR_DB_PATH,
            "index_files_exist": False,
            "document_count": 0,
            "issues": []
        }
        
        try:
            if not os.path.exists(VECTOR_DB_PATH):
                status["issues"].append(f"向量數據庫目錄不存在: {VECTOR_DB_PATH}")
                return status
            
            status["exists"] = True
            
            # 檢查索引文件
            index_faiss = os.path.join(VECTOR_DB_PATH, "index.faiss")
            index_pkl = os.path.join(VECTOR_DB_PATH, "index.pkl")
            
            if os.path.exists(index_faiss) and os.path.exists(index_pkl):
                status["index_files_exist"] = True
                
                # 嘗試載入並檢查文檔數量
                try:
                    indexer = DocumentIndexer()
                    vector_store = indexer.get_vector_store()
                    if vector_store and hasattr(vector_store, 'docstore'):
                        status["document_count"] = len(vector_store.docstore._dict)
                except Exception as e:
                    status["issues"].append(f"載入向量數據庫時出錯: {str(e)}")
            else:
                status["issues"].append("向量數據庫索引文件不存在")
            
            # 檢查索引記錄
            indexed_files_path = os.path.join(VECTOR_DB_PATH, "indexed_files.pkl")
            if not os.path.exists(indexed_files_path):
                status["issues"].append("索引記錄文件不存在")
            
        except Exception as e:
            status["issues"].append(f"檢查向量數據庫時出錯: {str(e)}")
        
        return status
    
    def _check_file_encoding_issues(self) -> Dict[str, Any]:
        """檢查文件編碼問題"""
        logger.info("檢查文件編碼問題...")
        
        status = {
            "total_checked": 0,
            "encoding_issues": 0,
            "problematic_files": [],
            "recommendations": []
        }
        
        try:
            # 檢查一些樣本文件
            sample_files = []
            for root, dirs, files in os.walk(Q_DRIVE_PATH):
                for file in files[:5]:  # 每個目錄最多檢查5個文件
                    ext = os.path.splitext(file)[1].lower()
                    if ext in ['.txt', '.md', '.csv']:
                        sample_files.append(os.path.join(root, file))
                        if len(sample_files) >= 20:  # 最多檢查20個文件
                            break
                if len(sample_files) >= 20:
                    break
            
            status["total_checked"] = len(sample_files)
            
            for file_path in sample_files:
                try:
                    # 嘗試用UTF-8讀取
                    with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
                        f.read(1024)  # 讀取前1KB
                except UnicodeDecodeError:
                    status["encoding_issues"] += 1
                    status["problematic_files"].append(file_path)
                except Exception:
                    continue
            
            if status["encoding_issues"] > 0:
                status["recommendations"].append("建議啟用自動編碼檢測功能")
                status["recommendations"].append("考慮使用chardet庫進行編碼檢測")
            
        except Exception as e:
            logger.error(f"檢查文件編碼時出錯: {str(e)}")
        
        return status
    
    def _check_index_integrity(self) -> Dict[str, Any]:
        """檢查索引完整性"""
        logger.info("檢查索引完整性...")
        
        status = {
            "indexed_files": 0,
            "missing_files": 0,
            "outdated_files": 0,
            "issues": []
        }
        
        try:
            indexer = DocumentIndexer()
            indexed_files = indexer.indexed_files
            
            status["indexed_files"] = len(indexed_files)
            
            # 檢查已索引文件是否仍存在
            for file_path, mtime in indexed_files.items():
                if not os.path.exists(file_path):
                    status["missing_files"] += 1
                else:
                    current_mtime = os.path.getmtime(file_path)
                    if current_mtime > mtime:
                        status["outdated_files"] += 1
            
            if status["missing_files"] > 0:
                status["issues"].append(f"{status['missing_files']} 個已索引文件不存在")
            
            if status["outdated_files"] > 0:
                status["issues"].append(f"{status['outdated_files']} 個已索引文件已過期")
            
        except Exception as e:
            status["issues"].append(f"檢查索引完整性時出錯: {str(e)}")
        
        return status
    
    def _check_dynamic_rag_scan(self) -> Dict[str, Any]:
        """檢查動態RAG文件掃描"""
        logger.info("檢查動態RAG文件掃描...")
        
        status = {
            "scan_successful": False,
            "files_found": 0,
            "scan_time": 0,
            "issues": []
        }
        
        try:
            from rag_engine.dynamic_rag_base import SmartFileRetriever
            
            start_time = time.time()
            retriever = SmartFileRetriever()
            
            # 測試文件檢索
            test_query = "測試查詢"
            relevant_files = retriever.retrieve_relevant_files(test_query, max_files=10)
            
            status["scan_time"] = time.time() - start_time
            status["files_found"] = len(relevant_files)
            status["scan_successful"] = True
            
            if len(relevant_files) == 0:
                status["issues"].append("動態RAG未找到任何相關文件")
            
        except Exception as e:
            status["issues"].append(f"動態RAG掃描測試失敗: {str(e)}")
        
        return status
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """分析性能"""
        logger.info("分析系統性能...")
        
        analysis = {
            "memory_usage": "unknown",
            "disk_space": "unknown",
            "recommendations": []
        }
        
        try:
            import psutil
            
            # 內存使用情況
            memory = psutil.virtual_memory()
            analysis["memory_usage"] = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent
            }
            
            if memory.percent > 80:
                analysis["recommendations"].append("系統內存使用率過高，建議減少批處理大小")
            
            # 磁盤空間
            disk = psutil.disk_usage(VECTOR_DB_PATH)
            analysis["disk_space"] = {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 2)
            }
            
            if disk.free < 1024**3:  # 少於1GB
                analysis["recommendations"].append("磁盤空間不足，建議清理或擴展存儲")
            
        except ImportError:
            analysis["recommendations"].append("建議安裝psutil庫以獲得更詳細的性能分析")
        except Exception as e:
            logger.error(f"性能分析時出錯: {str(e)}")
        
        return analysis
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成修復建議"""
        recommendations = []
        
        # Q槽問題建議
        if not results["q_drive_status"]["accessible"]:
            recommendations.append("檢查Q槽網絡連接和權限設置")
        
        # 向量數據庫問題建議
        if not results["vector_db_status"]["index_files_exist"]:
            recommendations.append("需要重新建立向量數據庫索引")
        
        # 編碼問題建議
        if results["encoding_issues"]["encoding_issues"] > 0:
            recommendations.append("啟用自動編碼檢測功能")
            recommendations.append("安裝chardet庫: pip install chardet")
        
        # 索引完整性問題建議
        if results["index_integrity"]["missing_files"] > 0:
            recommendations.append("清理無效的索引記錄")
        
        if results["index_integrity"]["outdated_files"] > 0:
            recommendations.append("更新過期的文件索引")
        
        # 動態RAG問題建議
        if not results["dynamic_scan_status"]["scan_successful"]:
            recommendations.append("檢查動態RAG配置和文件權限")
        
        # 性能優化建議
        perf = results["performance_analysis"]
        if isinstance(perf, dict) and "recommendations" in perf:
            recommendations.extend(perf["recommendations"])
        
        return recommendations
    
    def _determine_system_status(self, results: Dict[str, Any]) -> str:
        """確定系統狀態"""
        critical_issues = 0
        minor_issues = 0
        
        # 檢查關鍵問題
        if not results["q_drive_status"]["accessible"]:
            critical_issues += 1
        
        if not results["vector_db_status"]["index_files_exist"]:
            critical_issues += 1
        
        if not results["dynamic_scan_status"]["scan_successful"]:
            minor_issues += 1
        
        if results["encoding_issues"]["encoding_issues"] > 5:
            minor_issues += 1
        
        if critical_issues > 0:
            return "critical"
        elif minor_issues > 2:
            return "warning"
        else:
            return "healthy"
    
    def auto_fix_issues(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """自動修復問題"""
        logger.info("開始自動修復...")
        
        fix_results = {
            "fixes_attempted": 0,
            "fixes_successful": 0,
            "fixes_failed": 0,
            "details": []
        }
        
        # 修復索引完整性問題
        if results["index_integrity"]["missing_files"] > 0:
            fix_results["fixes_attempted"] += 1
            try:
                self._fix_missing_index_files()
                fix_results["fixes_successful"] += 1
                fix_results["details"].append("已清理無效索引記錄")
            except Exception as e:
                fix_results["fixes_failed"] += 1
                fix_results["details"].append(f"清理索引記錄失敗: {str(e)}")
        
        # 應用優化配置
        fix_results["fixes_attempted"] += 1
        try:
            from config.rag_optimization import apply_rag_optimizations
            apply_rag_optimizations()
            fix_results["fixes_successful"] += 1
            fix_results["details"].append("已應用RAG優化配置")
        except Exception as e:
            fix_results["fixes_failed"] += 1
            fix_results["details"].append(f"應用優化配置失敗: {str(e)}")
        
        return fix_results
    
    def _fix_missing_index_files(self):
        """修復缺失的索引文件"""
        indexer = DocumentIndexer()
        indexed_files = indexer.indexed_files.copy()
        
        removed_count = 0
        for file_path in list(indexed_files.keys()):
            if not os.path.exists(file_path):
                del indexer.indexed_files[file_path]
                removed_count += 1
        
        if removed_count > 0:
            indexer._save_indexed_files()
            logger.info(f"已清理 {removed_count} 個無效索引記錄")

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG系統診斷和修復工具")
    parser.add_argument("--auto-fix", action="store_true", help="自動修復發現的問題")
    parser.add_argument("--output", type=str, help="診斷結果輸出文件路徑")
    
    args = parser.parse_args()
    
    # 創建診斷工具
    diagnostic = RAGDiagnosticTool()
    
    # 運行診斷
    results = diagnostic.run_full_diagnostic()
    
    # 自動修復
    if args.auto_fix:
        fix_results = diagnostic.auto_fix_issues(results)
        results["auto_fix_results"] = fix_results
    
    # 輸出結果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"診斷結果已保存到: {args.output}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    
    # 顯示摘要
    print(f"\n=== RAG系統診斷摘要 ===")
    print(f"系統狀態: {results['system_status']}")
    print(f"Q槽訪問: {'正常' if results['q_drive_status']['accessible'] else '異常'}")
    print(f"向量數據庫: {'正常' if results['vector_db_status']['index_files_exist'] else '異常'}")
    print(f"編碼問題: {results['encoding_issues']['encoding_issues']} 個文件")
    print(f"建議數量: {len(results['recommendations'])}")
    
    if results['recommendations']:
        print(f"\n=== 修復建議 ===")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")

if __name__ == "__main__":
    main()