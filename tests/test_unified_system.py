#!/usr/bin/env python
"""
統一系統測試 - 測試新的平台管理和設置流程
"""

import os
import sys
import unittest
import requests
import time
from pathlib import Path

# 添加項目根目錄到路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.platform_manager import get_platform_manager, PlatformType
from utils.setup_flow_manager import get_setup_flow_manager, SetupStep, RAGMode

class TestPlatformManager(unittest.TestCase):
    """測試平台管理器"""
    
    def setUp(self):
        self.platform_manager = get_platform_manager()
    
    def test_get_available_platforms(self):
        """測試獲取可用平台"""
        platforms = self.platform_manager.get_available_platforms()
        self.assertIsInstance(platforms, list)
        self.assertGreater(len(platforms), 0)
        
        # 檢查必要字段
        for platform in platforms:
            self.assertIn("type", platform)
            self.assertIn("name", platform)
            self.assertIn("status", platform)
    
    def test_set_platform(self):
        """測試設置平台"""
        # 測試有效平台
        result = self.platform_manager.set_platform("huggingface")
        self.assertTrue(result)
        
        # 測試無效平台
        result = self.platform_manager.set_platform("invalid_platform")
        self.assertFalse(result)
    
    def test_get_available_models(self):
        """測試獲取可用模型"""
        self.platform_manager.set_platform("huggingface")
        models = self.platform_manager.get_available_models()
        
        self.assertIn("language_models", models)
        self.assertIn("embedding_models", models)
        self.assertIsInstance(models["language_models"], list)
        self.assertIsInstance(models["embedding_models"], list)

class TestSetupFlowManager(unittest.TestCase):
    """測試設置流程管理器"""
    
    def setUp(self):
        self.setup_manager = get_setup_flow_manager()
        # 重置設置以確保測試環境乾淨
        self.setup_manager.reset_setup()
    
    def test_initial_state(self):
        """測試初始狀態"""
        self.assertFalse(self.setup_manager.is_setup_completed())
        self.assertEqual(self.setup_manager.get_current_step(), SetupStep.PLATFORM_SELECTION.value)
    
    def test_platform_selection_flow(self):
        """測試平台選擇流程"""
        # 獲取平台選擇數據
        data = self.setup_manager.get_platform_selection_data()
        self.assertEqual(data["step"], SetupStep.PLATFORM_SELECTION.value)
        self.assertIn("platforms", data)
        
        # 設置平台
        result = self.setup_manager.set_platform("huggingface")
        self.assertTrue(result["success"])
        self.assertEqual(self.setup_manager.get_current_step(), SetupStep.MODEL_SELECTION.value)
    
    def test_model_selection_flow(self):
        """測試模型選擇流程"""
        # 先設置平台
        self.setup_manager.set_platform("huggingface")
        
        # 獲取模型選擇數據
        data = self.setup_manager.get_model_selection_data()
        self.assertEqual(data["step"], SetupStep.MODEL_SELECTION.value)
        self.assertIn("models", data)
        
        # 設置模型
        result = self.setup_manager.set_models(
            "Qwen/Qwen2-0.5B-Instruct",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.assertTrue(result["success"])
        self.assertEqual(self.setup_manager.get_current_step(), SetupStep.RAG_MODE_SELECTION.value)
    
    def test_rag_mode_selection_flow(self):
        """測試 RAG 模式選擇流程"""
        # 先完成前面的步驟
        self.setup_manager.set_platform("huggingface")
        self.setup_manager.set_models(
            "Qwen/Qwen2-0.5B-Instruct",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        # 獲取 RAG 模式選擇數據
        data = self.setup_manager.get_rag_mode_selection_data()
        self.assertEqual(data["step"], SetupStep.RAG_MODE_SELECTION.value)
        self.assertIn("rag_modes", data)
        
        # 設置 RAG 模式
        result = self.setup_manager.set_rag_mode(RAGMode.TRADITIONAL.value)
        self.assertTrue(result["success"])
        self.assertEqual(self.setup_manager.get_current_step(), SetupStep.CONFIGURATION_REVIEW.value)
    
    def test_complete_setup_flow(self):
        """測試完整設置流程"""
        # 完成所有步驟
        self.setup_manager.set_platform("huggingface")
        self.setup_manager.set_models(
            "Qwen/Qwen2-0.5B-Instruct",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.setup_manager.set_rag_mode(RAGMode.TRADITIONAL.value)
        
        # 完成設置
        result = self.setup_manager.complete_setup()
        self.assertTrue(result["success"])
        self.assertTrue(self.setup_manager.is_setup_completed())
        self.assertEqual(self.setup_manager.get_current_step(), SetupStep.READY.value)

class TestAPIEndpoints(unittest.TestCase):
    """測試 API 端點"""
    
    @classmethod
    def setUpClass(cls):
        cls.api_url = "http://localhost:8000"
        # 等待 API 服務啟動
        cls._wait_for_api()
    
    @classmethod
    def _wait_for_api(cls, timeout=30):
        """等待 API 服務啟動"""
        for _ in range(timeout):
            try:
                response = requests.get(f"{cls.api_url}/")
                if response.status_code == 200:
                    return
            except:
                pass
            time.sleep(1)
        raise Exception("API 服務未啟動")
    
    def test_setup_status_endpoint(self):
        """測試設置狀態端點"""
        response = requests.get(f"{self.api_url}/api/setup/status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("setup_completed", data)
        self.assertIn("current_step", data)
        self.assertIn("progress", data)
    
    def test_platforms_endpoint(self):
        """測試平台列表端點"""
        response = requests.get(f"{self.api_url}/api/setup/platforms")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("platforms", data)
        self.assertIsInstance(data["platforms"], list)
    
    def test_platform_selection_endpoint(self):
        """測試平台選擇端點"""
        response = requests.post(
            f"{self.api_url}/api/setup/platform",
            json={"platform_type": "huggingface"}
        )
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])

if __name__ == "__main__":
    # 創建測試套件
    suite = unittest.TestSuite()
    
    # 添加測試
    suite.addTest(unittest.makeSuite(TestPlatformManager))
    suite.addTest(unittest.makeSuite(TestSetupFlowManager))
    
    # 只有在 API 服務運行時才測試 API 端點
    try:
        requests.get("http://localhost:8000/", timeout=2)
        suite.addTest(unittest.makeSuite(TestAPIEndpoints))
        print("✅ 包含 API 端點測試")
    except:
        print("⚠️ API 服務未運行，跳過 API 端點測試")
    
    # 運行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回結果
    sys.exit(0 if result.wasSuccessful() else 1)
