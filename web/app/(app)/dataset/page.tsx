
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { useAuth } from "@/app/_components/AuthProvider";
import DataTable from "@/app/_components/DataTable";
import StatCard from "@/app/_components/StatCard";
import Tabs from "@/app/_components/Tabs";
import { DEMO_PUBLIC_READONLY, DEMO_READONLY_MESSAGE } from "@/lib/demo";

type UploadResponse = {
  version_id: string;
  original_filename: string;
  stored_filename: string;
  filename: string;
  bytes: number;
  row_count?: number | null;
  columns: string[];
  preview: Record<string, string | number | null>[];
};

type DatasetItem = {
  version_id: string;
  original_filename: string;
  stored_filename: string;
  size_bytes: number;
  uploaded_at: string;
  schema: string[];
  row_count?: number | null;
  is_active: boolean;
};

type DatasetPreview = {
  version_id: string;
  original_filename: string;
  stored_filename: string;
  bytes: number;
  columns: string[];
  rows: Record<string, string | number | null | string[]>[];
  row_count?: number | null;
};

type ScoredPreviewSummary = {
  fraud: number;
  legit: number;
  fraud_rate: number;
  avg_score: number;
};

type ScoredPreview = DatasetPreview & {
  summary?: ScoredPreviewSummary;
};

type DatasetProfile = {
  version_id: string;
  sample_rows: number;
  columns: string[];
  missing_by_column: Record<string, number>;
  duplicate_estimate_percent: number;
  numeric_stats: Record<
    string,
    { min: number; max: number; mean: number; p95: number }
  >;
  invalid_timestamp_count: number;
  generated_at?: string;
  mapping?: SchemaMapping | null;
  mapping_source?: "custom" | "auto";
};

type SchemaMapping = {
  amount_col?: string | null;
  timestamp_col?: string | null;
  user_id_col?: string | null;
  merchant_col?: string | null;
  device_id_col?: string | null;
  country_col?: string | null;
  label_col?: string | null;
};

type ScoringJob = {
  job_id: string;
  dataset_version_id: string;
  status: string;
  rows_done: number;
  rows_total?: number | null;
  output_path?: string | null;
  output_format?: string;
  threshold?: number;
  model_version?: string | null;
  created_at: string;
  updated_at: string;
  error?: string;
};

type ScoringResults = {
  job_id: string;
  offset: number;
  limit: number;
  has_more: boolean;
  rows: Record<string, string | number | null | string[]>[];
  is_partial?: boolean;
  rows_available?: number;
  job_status?: string;
};

type AuditEntry = {
  id: string;
  timestamp: string;
  actor?: string | null;
  action: string;
  decision: string;
  model_name: string;
  model_version: string;
  score: number;
};

const API_BASE = "/api/proxy";

const buildLoginUrl = () => {
  if (typeof window === "undefined") {
    return "/login";
  }
  const next = `${window.location.pathname}${window.location.search}`;
  return `/login?next=${encodeURIComponent(next)}`;
};

const redirectToLogin = () => {
  if (typeof window === "undefined") {
    return;
  }
  window.location.href = buildLoginUrl();
};

const parseErrorMessage = (raw: string) => {
  if (!raw) {
    return "Request failed. Please try again.";
  }
  try {
    const parsed = JSON.parse(raw) as { detail?: string };
    if (parsed && typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch {
    // fall through
  }
  return raw;
};

const formatBytes = (bytes?: number | null) => {
  if (!bytes) {
    return "n/a";
  }
  const mb = bytes / (1024 * 1024);
  return `${Math.round(mb)} MB`;
};

const toTableRows = (
  rows: Record<string, string | number | null | string[]>[],
  columns: string[]
) =>
  rows.map((row, rowIndex) =>
    columns.map((column, colIndex) => (
      <span key={`${rowIndex}-${colIndex}`}>{String(row[column] ?? "")}</span>
    ))
  );

export default function DatasetPage() {
  const [activeTab, setActiveTab] = useState("upload");
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [resp, setResp] = useState<UploadResponse | null>(null);
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [authRequired, setAuthRequired] = useState(false);
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState("");
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [activeVersionId, setActiveVersionId] = useState<string | null>(null);
  const { user } = useAuth();
  const demoWriteBlocked = DEMO_PUBLIC_READONLY && !user;
  const demoWriteTitle = demoWriteBlocked ? DEMO_READONLY_MESSAGE : undefined;

  const ensureWriteAccess = () => {
    if (!demoWriteBlocked) {
      return true;
    }
    redirectToLogin();
    return false;
  };

  const [previewMode, setPreviewMode] = useState<"raw" | "scored">("raw");
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [scoredPreview, setScoredPreview] = useState<ScoredPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [scoredThreshold, setScoredThreshold] = useState(0.5);
  const [scoredLimit, setScoredLimit] = useState(200);
  const [scoredFraudOnly, setScoredFraudOnly] = useState(false);

  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [schemaMapping, setSchemaMapping] = useState<SchemaMapping>({
    amount_col: null,
    timestamp_col: null,
    user_id_col: null,
    merchant_col: null,
    device_id_col: null,
    country_col: null,
    label_col: null
  });
  const [mappingSaving, setMappingSaving] = useState(false);
  const [mappingError, setMappingError] = useState<string | null>(null);
  const [mappingMessage, setMappingMessage] = useState<string | null>(null);

  const [jobs, setJobs] = useState<ScoringJob[]>([]);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [jobThreshold, setJobThreshold] = useState(0.5);
  const [jobModelVersion, setJobModelVersion] = useState("");
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const [results, setResults] = useState<ScoringResults | null>(null);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [resultsError, setResultsError] = useState<string | null>(null);
  const [resultsOffset, setResultsOffset] = useState(0);
  const [resultsLimit, setResultsLimit] = useState(200);
  const [fraudOnly, setFraudOnly] = useState(true);
  const [resultsThreshold, setResultsThreshold] = useState(0.5);
  const [selectedTxIds, setSelectedTxIds] = useState<Set<string>>(new Set());
  const [caseMessage, setCaseMessage] = useState<string | null>(null);
  const [caseError, setCaseError] = useState<string | null>(null);
  const [createdCaseId, setCreatedCaseId] = useState<string | null>(null);
  const [feedbackLabel, setFeedbackLabel] = useState<"FRAUD" | "LEGIT">("FRAUD");
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);

  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditError, setAuditError] = useState<string | null>(null);

  const handleAuthFailure = () => {
    setAuthRequired(true);
    setDatasetError("Please login to continue.");
    redirectToLogin();
  };

  const handleAuthResponse = (res: Response) => {
    if (res.status === 401) {
      handleAuthFailure();
      return true;
    }
    return false;
  };

  const fetchDatasets = useCallback(async () => {
    setDatasetsLoading(true);
    setDatasetError(null);
    setAuthRequired(false);
    try {
      const res = await fetch(`${API_BASE}/datasets`, { credentials: "include" });
      if (handleAuthResponse(res)) {
        return;
      }
      if (res.status === 403) {
        setDatasetError("You do not have permission to view datasets.");
        return;
      }
      if (!res.ok) {
        setDatasetError(parseErrorMessage(await res.text()));
        return;
      }
      const data = (await res.json()) as DatasetItem[];
      setDatasets(data);
      const activeItem = data.find((item) => item.is_active);
      setActiveVersionId(activeItem?.version_id ?? null);
      setSelectedVersionId((current) => current ?? activeItem?.version_id ?? data[0]?.version_id ?? null);
    } catch {
      setDatasetError("Failed to load datasets.");
    } finally {
      setDatasetsLoading(false);
    }
  }, []);

  const setActiveDataset = async (versionId: string) => {
    if (!ensureWriteAccess()) {
      return;
    }
    setDatasetError(null);
    try {
      const res = await fetch(`${API_BASE}/datasets/set-active`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version_id: versionId })
      });
      if (handleAuthResponse(res)) {
        return;
      }
      if (res.status === 403) {
        setDatasetError("You do not have permission to update datasets.");
        return;
      }
      if (!res.ok) {
        setDatasetError(parseErrorMessage(await res.text()));
        return;
      }
      await fetchDatasets();
    } catch {
      setDatasetError("Failed to set active dataset.");
    }
  };

  const deleteDataset = async (versionId: string, label: string) => {
    if (!ensureWriteAccess()) {
      return;
    }
    if (!confirm(`Delete dataset: ${label}?`)) {
      return;
    }
    setDatasetError(null);
    try {
      const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(versionId)}`, {
        method: "DELETE",
        credentials: "include"
      });
      if (handleAuthResponse(res)) {
        return;
      }
      if (res.status === 403) {
        setDatasetError("You do not have permission to delete datasets.");
        return;
      }
      if (!res.ok) {
        setDatasetError(parseErrorMessage(await res.text()));
        return;
      }
      if (selectedVersionId === versionId) {
        setSelectedVersionId(null);
        setPreview(null);
        setScoredPreview(null);
      }
      await fetchDatasets();
    } catch {
      setDatasetError("Failed to delete dataset.");
    }
  };

  const downloadDataset = (versionId: string) => {
    const url = `${API_BASE}/datasets/${encodeURIComponent(versionId)}/download`;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const upload = async () => {
    if (!ensureWriteAccess()) {
      return;
    }
    if (!file || busy) {
      return;
    }
    setBusy(true);
    setDatasetError(null);
    setResp(null);
    setProgress(0);

    const fd = new FormData();
    fd.append("file", file);

    await new Promise<void>((resolve) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/datasets/upload`);
      xhr.withCredentials = true;

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          setProgress(Math.round((event.loaded / event.total) * 100));
        }
      };

      xhr.onload = () => {
        setBusy(false);
        if (xhr.status === 401) {
          handleAuthFailure();
          resolve();
          return;
        }
        if (xhr.status === 403) {
          setDatasetError("You do not have permission to upload datasets.");
          resolve();
          return;
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const parsed = JSON.parse(xhr.responseText) as UploadResponse;
            setResp(parsed);
            void fetchDatasets();
          } catch {
            setDatasetError("Upload completed, but response could not be parsed.");
          }
        } else {
          setDatasetError(parseErrorMessage(xhr.responseText));
        }
        resolve();
      };

      xhr.onerror = () => {
        setBusy(false);
        setDatasetError("Network error during upload.");
        resolve();
      };

      xhr.send(fd);
    });
  };

  const fetchPreview = useCallback(
    async (versionId: string) => {
      setPreviewLoading(true);
      setPreviewError(null);
      try {
        const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(versionId)}/preview?limit=50`, {
          credentials: "include"
        });
        if (handleAuthResponse(res)) {
          return;
        }
        if (res.status === 403) {
          setPreviewError("You do not have permission to view dataset previews.");
          return;
        }
        if (!res.ok) {
          setPreviewError(parseErrorMessage(await res.text()));
          return;
        }
        const data = (await res.json()) as DatasetPreview;
        setPreview(data);
      } catch {
        setPreviewError("Failed to load dataset preview.");
      } finally {
        setPreviewLoading(false);
      }
    },
    []
  );

  const fetchScoredPreview = useCallback(
    async (versionId: string, threshold: number, limit: number) => {
      setPreviewLoading(true);
      setPreviewError(null);
      try {
        const res = await fetch(
          `${API_BASE}/datasets/${encodeURIComponent(versionId)}/scored-preview?limit=${limit}&threshold=${threshold}`,
          { credentials: "include" }
        );
        if (handleAuthResponse(res)) {
          return;
        }
        if (res.status === 403) {
          setPreviewError("You do not have permission to view scored previews.");
          return;
        }
        if (!res.ok) {
          setPreviewError(parseErrorMessage(await res.text()));
          return;
        }
        const data = (await res.json()) as ScoredPreview;
        setScoredPreview(data);
      } catch {
        setPreviewError("Failed to load scored preview.");
      } finally {
        setPreviewLoading(false);
      }
    },
    []
  );

  const fetchProfile = useCallback(
    async (versionId: string) => {
      setProfileLoading(true);
      setProfileError(null);
      try {
        const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(versionId)}/profile`, {
          credentials: "include"
        });
        if (handleAuthResponse(res)) {
          return;
        }
        if (res.status === 403) {
          setProfileError("You do not have permission to view dataset profiles.");
          return;
        }
        if (!res.ok) {
          setProfileError(parseErrorMessage(await res.text()));
          return;
        }
        const data = (await res.json()) as DatasetProfile;
        setProfile(data);
        setSchemaMapping(data.mapping ?? {});
      } catch {
        setProfileError("Failed to load profile.");
      } finally {
        setProfileLoading(false);
      }
    },
    []
  );

  const saveSchemaMapping = async () => {
    if (!ensureWriteAccess()) {
      return;
    }
    if (!selectedVersionId) {
      setMappingError("Select a dataset first.");
      return;
    }
    setMappingSaving(true);
    setMappingError(null);
    setMappingMessage(null);
    try {
      const res = await fetch(
        `${API_BASE}/datasets/${encodeURIComponent(selectedVersionId)}/schema-mapping`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(schemaMapping)
        }
      );
      if (handleAuthResponse(res)) {
        return;
      }
      if (res.status === 403) {
        setMappingError("You do not have permission to update schema mappings.");
        return;
      }
      if (!res.ok) {
        setMappingError(parseErrorMessage(await res.text()));
        return;
      }
      setMappingMessage("Mapping saved.");
      void fetchProfile(selectedVersionId);
    } catch {
      setMappingError("Failed to save mapping.");
    } finally {
      setMappingSaving(false);
    }
  };

  const fetchJobs = useCallback(async () => {
    setJobsLoading(true);
    setJobsError(null);
    try {
      const res = await fetch(`${API_BASE}/scoring-jobs`, { credentials: "include" });
      if (handleAuthResponse(res)) {
        return;
      }
      if (res.status === 403) {
        setJobsError("You do not have permission to view scoring jobs.");
        return;
      }
      if (!res.ok) {
        setJobsError(parseErrorMessage(await res.text()));
        return;
      }
      const data = (await res.json()) as ScoringJob[];
      setJobs(data);
      const doneJob = data.find((job) => job.status === "done");
      setSelectedJobId((current) => current ?? doneJob?.job_id ?? data[0]?.job_id ?? null);
    } catch {
      setJobsError("Failed to load scoring jobs.");
    } finally {
      setJobsLoading(false);
    }
  }, []);

  const startJob = async () => {
    if (!ensureWriteAccess()) {
      return;
    }
    if (!selectedVersionId) {
      setJobsError("Select a dataset to score.");
      return;
    }
    setJobsError(null);
    try {
      const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(selectedVersionId)}/score`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          threshold: jobThreshold,
          model_version: jobModelVersion || undefined
        })
      });
      if (handleAuthResponse(res)) {
        return;
      }
      if (res.status === 403) {
        setJobsError("You do not have permission to start scoring jobs.");
        return;
      }
      if (!res.ok) {
        setJobsError(parseErrorMessage(await res.text()));
        return;
      }
      await fetchJobs();
      setActiveTab("jobs");
    } catch {
      setJobsError("Failed to start scoring job.");
    }
  };

  const fetchResults = useCallback(
    async (jobId: string, offset: number, limit: number, onlyFraud: boolean) => {
      setResultsLoading(true);
      setResultsError(null);
      const prediction = onlyFraud ? "&prediction=fraud" : "";
      try {
        const res = await fetch(
          `${API_BASE}/scoring-jobs/${encodeURIComponent(jobId)}/results?offset=${offset}&limit=${limit}${prediction}`,
          { credentials: "include" }
        );
        if (handleAuthResponse(res)) {
          return;
        }
        if (res.status === 403) {
          setResultsError("You do not have permission to view scoring results.");
          return;
        }
        if (!res.ok) {
          setResultsError(parseErrorMessage(await res.text()));
          return;
        }
        const data = (await res.json()) as ScoringResults;
        setResults(data);
      } catch {
        setResultsError("Failed to load results.");
      } finally {
        setResultsLoading(false);
      }
    },
    []
  );

  const createCases = async () => {
    if (!ensureWriteAccess()) {
      return;
    }
    if (!selectedJobId) {
      setCaseError("Select a scoring job first.");
      return;
    }
    if (selectedTxIds.size === 0) {
      setCaseError("Select at least one row.");
      return;
    }
    setCaseMessage(null);
    setCaseError(null);
    setCreatedCaseId(null);
    try {
      const res = await fetch(`${API_BASE}/cases/from-job/${encodeURIComponent(selectedJobId)}`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tx_ids: Array.from(selectedTxIds) })
      });
      if (handleAuthResponse(res)) {
        return;
      }
      if (res.status === 403) {
        setCaseError("You do not have permission to create cases.");
        return;
      }
      if (!res.ok) {
        setCaseError(parseErrorMessage(await res.text()));
        return;
      }
      const data = (await res.json()) as { created: number; case_id?: string };
      setCaseMessage(`Created ${data.created} case items.`);
      setCreatedCaseId(data.case_id ?? null);
      setSelectedTxIds(new Set());
    } catch {
      setCaseError("Failed to create cases.");
    }
  };

  const submitFeedback = async () => {
    if (!ensureWriteAccess()) {
      return;
    }
    if (selectedTxIds.size === 0) {
      setFeedbackError("Select at least one row.");
      return;
    }
    setFeedbackMessage(null);
    setFeedbackError(null);
    try {
      await Promise.all(
        Array.from(selectedTxIds).map(async (txId) => {
          const res = await fetch(`${API_BASE}/feedback`, {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              tx_id: txId,
              label: feedbackLabel,
              source: "analyst",
              job_id: selectedJobId ?? undefined
            })
          });
          if (res.status === 401) {
            handleAuthFailure();
            throw new Error("Missing credentials");
          }
          if (res.status === 403) {
            throw new Error("You do not have permission to submit feedback.");
          }
          if (!res.ok) {
            throw new Error(await res.text());
          }
        })
      );
      setFeedbackMessage("Feedback submitted.");
    } catch (err) {
      setFeedbackError(parseErrorMessage(String(err)));
    }
  };

  const submitRowFeedback = async (txId: string, label: "FRAUD" | "LEGIT") => {
    if (!ensureWriteAccess()) {
      return;
    }
    setFeedbackMessage(null);
    setFeedbackError(null);
    try {
      const res = await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tx_id: txId,
          label,
          source: "analyst",
          job_id: selectedJobId ?? undefined
        })
      });
      if (res.status === 401) {
        handleAuthFailure();
        return;
      }
      if (res.status === 403) {
        setFeedbackError("You do not have permission to submit feedback.");
        return;
      }
      if (!res.ok) {
        setFeedbackError(parseErrorMessage(await res.text()));
        return;
      }
      setFeedbackMessage(`Feedback saved for ${txId}.`);
    } catch {
      setFeedbackError("Failed to submit feedback.");
    }
  };
  const fetchAudit = useCallback(
    async (versionId?: string | null) => {
      setAuditLoading(true);
      setAuditError(null);
      try {
      const url = versionId
        ? `${API_BASE}/audit?entity_type=dataset&id=${encodeURIComponent(versionId)}&limit=100`
        : `${API_BASE}/audit?limit=100`;
      const res = await fetch(url, { credentials: "include" });
      if (handleAuthResponse(res)) {
        return;
      }
      if (res.status === 403) {
        setAuditError("You do not have permission to view audit logs.");
        return;
      }
      if (!res.ok) {
        setAuditError(parseErrorMessage(await res.text()));
        return;
      }
        const data = (await res.json()) as { items: AuditEntry[] };
        setAuditEntries(data.items ?? []);
      } catch {
        setAuditError("Failed to load audit logs.");
      } finally {
        setAuditLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    const fileParam = params.get("file");
    if (fileParam) {
      setSelectedVersionId(fileParam);
      setActiveTab("preview");
    }
  }, []);

  useEffect(() => {
    if (!selectedVersionId) {
      return;
    }
    if (activeTab === "preview") {
      if (previewMode === "raw") {
        void fetchPreview(selectedVersionId);
      } else {
        void fetchScoredPreview(selectedVersionId, scoredThreshold, scoredLimit);
      }
    }
    if (activeTab === "quality") {
      void fetchProfile(selectedVersionId);
    }
  }, [
    selectedVersionId,
    previewMode,
    scoredThreshold,
    scoredLimit,
    activeTab,
    fetchPreview,
    fetchScoredPreview,
    fetchProfile
  ]);

  useEffect(() => {
    if (activeTab === "jobs") {
      fetchJobs();
      const interval = setInterval(() => {
        fetchJobs();
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [activeTab, fetchJobs]);

  useEffect(() => {
    if (activeTab === "results" || activeTab === "cases") {
      fetchJobs();
    }
  }, [activeTab, fetchJobs]);

  useEffect(() => {
    if (activeTab === "results" && selectedJobId) {
      void fetchResults(selectedJobId, resultsOffset, resultsLimit, fraudOnly);
    }
  }, [activeTab, selectedJobId, resultsOffset, resultsLimit, fraudOnly, fetchResults]);

  useEffect(() => {
    if (activeTab !== "results" || !selectedJobId) {
      return;
    }
    if (results?.job_status !== "running") {
      return;
    }
    const interval = setInterval(() => {
      fetchResults(selectedJobId, resultsOffset, resultsLimit, fraudOnly);
    }, 7000);
    return () => clearInterval(interval);
  }, [activeTab, selectedJobId, resultsOffset, resultsLimit, fraudOnly, results, fetchResults]);

  useEffect(() => {
    setResultsOffset(0);
  }, [selectedJobId, fraudOnly, resultsLimit]);

  useEffect(() => {
    if (activeTab === "audit") {
      void fetchAudit(selectedVersionId);
    }
  }, [activeTab, selectedVersionId, fetchAudit]);

  const filteredDatasets = datasets.filter(
    (dataset) =>
      dataset.stored_filename !== "active_dataset.json" &&
      dataset.original_filename.toLowerCase().includes(filter.toLowerCase())
  );

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.version_id === selectedVersionId) ?? null,
    [datasets, selectedVersionId]
  );

  const previewColumns = preview?.columns ?? [];
  const previewRows = preview?.rows ?? [];
  const scoredPreviewRows = scoredPreview?.rows ?? [];
  const scoredPreviewSummary = scoredPreview?.summary;

  const mappingWarnings = useMemo(() => {
    const missing: string[] = [];
    if (!schemaMapping.amount_col) missing.push("amount");
    if (!schemaMapping.timestamp_col) missing.push("timestamp");
    if (!schemaMapping.user_id_col) missing.push("user");
    if (!schemaMapping.merchant_col) missing.push("merchant");
    if (!schemaMapping.device_id_col) missing.push("device");
    if (!schemaMapping.country_col) missing.push("country");
    return missing;
  }, [schemaMapping]);

  const scoredPreviewDisplayRows = useMemo(() => {
    if (!scoredPreviewRows.length) {
      return [];
    }
    if (!scoredFraudOnly) {
      return scoredPreviewRows;
    }
    return scoredPreviewRows.filter(
      (row) => String(row._prediction ?? "").toUpperCase() === "FRAUD"
    );
  }, [scoredPreviewRows, scoredFraudOnly]);

  const scoredPreviewTableColumns = ["_prediction", "_risk_score", "_reasons"];

  const scoredPreviewTableRows = useMemo(
    () =>
      scoredPreviewDisplayRows.map((row, rowIndex) => {
        const prediction = String(row._prediction ?? "").toUpperCase();
        const pillStyle =
          prediction === "FRAUD"
            ? { background: "rgba(248, 113, 113, 0.2)", color: "#b91c1c", fontWeight: 600 }
            : { background: "rgba(34, 211, 238, 0.15)", color: "var(--accent)", fontWeight: 600 };
        const reasons = Array.isArray(row._reasons)
          ? row._reasons.join(", ")
          : Array.isArray(row.reason_codes)
          ? row.reason_codes.join(", ")
          : String(row._reasons ?? row.reason_codes ?? "");
        const score = row._risk_score ?? "";
        return [
          <span key={`pred-${rowIndex}`} className="pill" style={pillStyle}>
            {prediction || "n/a"}
          </span>,
          String(score ?? ""),
          reasons || ""
        ];
      }),
    [scoredPreviewDisplayRows]
  );

  const displayedResults = useMemo(() => {
    if (!results) {
      return [];
    }
    return results.rows.filter((row) => {
      const score = Number(row._risk_score ?? 0);
      return score >= resultsThreshold;
    });
  }, [results, resultsThreshold]);

  const txSelection = useMemo(() => new Set(selectedTxIds), [selectedTxIds]);

  const toggleTx = (txId: string) => {
    setSelectedTxIds((prev) => {
      const next = new Set(prev);
      if (next.has(txId)) {
        next.delete(txId);
      } else {
        next.add(txId);
      }
      return next;
    });
  };

  const clearSelection = () => setSelectedTxIds(new Set());

  const updateMapping = (key: keyof SchemaMapping, value: string) => {
    setSchemaMapping((prev) => ({
      ...prev,
      [key]: value || null
    }));
  };

  const riskBins = useMemo(() => {
    const buckets = [0, 0, 0, 0, 0];
    displayedResults.forEach((row) => {
      const score = Number(row._risk_score ?? 0);
      if (score >= 0.8) {
        buckets[4] += 1;
      } else if (score >= 0.6) {
        buckets[3] += 1;
      } else if (score >= 0.4) {
        buckets[2] += 1;
      } else if (score >= 0.2) {
        buckets[1] += 1;
      } else {
        buckets[0] += 1;
      }
    });
    return buckets;
  }, [displayedResults]);

  const topSegments = useMemo(() => {
    const segmentKeys = [
      { label: "Merchant", keys: ["merchant_id", "merchant", "merchant_name"] },
      { label: "Country", keys: ["country", "merchant_country", "billing_country"] },
      { label: "Device", keys: ["device", "device_id", "device_type"] }
    ];
    return segmentKeys.map((segment) => {
      const counts = new Map<string, number>();
      displayedResults.forEach((row) => {
        const key = segment.keys.find((item) => row[item] !== undefined);
        if (!key) {
          return;
        }
        const value = String(row[key] ?? "Unknown");
        counts.set(value, (counts.get(value) ?? 0) + 1);
      });
      const top = Array.from(counts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);
      return { label: segment.label, top };
    });
  }, [displayedResults]);

  const resultTableColumns = useMemo(() => {
    const base = ["Select", "Tx ID", "Risk", "Prediction"];
    const optional = ["amount", "merchant_id", "country", "device"];
    const available = new Set(
      displayedResults.flatMap((row) => Object.keys(row)).map((key) => key.toLowerCase())
    );
    const extras = optional.filter((key) => available.has(key));
    return [...base, ...extras, "Reason Codes", "Feedback"];
  }, [displayedResults]);

  const resultTableRows = displayedResults.map((row, index) => {
    const txId = String(row._tx_id ?? `row-${index}`);
    const risk = Number(row._risk_score ?? 0).toFixed(3);
    const prediction = String(row._prediction ?? "");
    const reasonCodes = Array.isArray(row.reason_codes)
      ? row.reason_codes.join(", ")
      : String(row.reason_codes ?? "");
    const amount = row.amount ?? row.Amount ?? row.transaction_amount ?? row.amt ?? "";
    const merchant = row.merchant_id ?? row.merchant ?? row.merchant_name ?? "";
    const country = row.country ?? row.merchant_country ?? row.billing_country ?? "";
    const device = row.device ?? row.device_id ?? row.device_type ?? "";

    const extras: Record<string, string | number | null> = {
      amount,
      merchant_id: merchant,
      country,
      device
    };

    return [
      <input
        key={`select-${txId}`}
        type="checkbox"
        checked={txSelection.has(txId)}
        onChange={() => toggleTx(txId)}
      />,
      txId,
      risk,
      prediction,
      ...(resultTableColumns.includes("amount") ? [String(extras.amount ?? "")] : []),
      ...(resultTableColumns.includes("merchant_id") ? [String(extras.merchant_id ?? "")] : []),
      ...(resultTableColumns.includes("country") ? [String(extras.country ?? "")] : []),
      ...(resultTableColumns.includes("device") ? [String(extras.device ?? "")] : []),
      reasonCodes,
      <div key={`feedback-${txId}`} style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
        <button
          className="button secondary"
          type="button"
          onClick={() => submitRowFeedback(txId, "FRAUD")}
          disabled={demoWriteBlocked}
          title={demoWriteTitle}
        >
          Confirm Fraud
        </button>
        <button
          className="button secondary"
          type="button"
          onClick={() => submitRowFeedback(txId, "LEGIT")}
          disabled={demoWriteBlocked}
          title={demoWriteTitle}
        >
          False Positive
        </button>
      </div>
    ];
  });

  const tabs = [
    { label: "Upload", value: "upload" },
    { label: "Library", value: "library" },
    { label: "Preview", value: "preview" },
    { label: "Quality/Profile", value: "quality" },
    { label: "Scoring Jobs", value: "jobs" },
    { label: "Fraud Results", value: "results" },
    { label: "Create Cases", value: "cases" },
    { label: "Audit", value: "audit" }
  ];
  return (
    <section className="section">
      <div className="page-header">
        <div className="page-header-text">
          <h3>Dataset Operations</h3>
          <p className="muted">Manage uploads, scoring workflows, and case creation in one workspace.</p>
        </div>
      </div>

      <Tabs tabs={tabs} active={activeTab} onChange={setActiveTab} />

      {activeTab === "upload" && (
        <>
          <div>
            <h3>Dataset Upload</h3>
            <p className="muted">Upload a CSV directly to the backend (supports large files).</p>
          </div>

          <div className="card">
            <div className="form">
              <label className="field">
                <span>CSV file</span>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                />
              </label>
              <div className="toolbar">
                <button
                  className="button"
                  type="button"
                  onClick={upload}
                  disabled={demoWriteBlocked || !file || busy}
                  title={demoWriteTitle}
                >
                  {busy ? `Uploading... ${progress}%` : "Upload"}
                </button>
                <button
                  className="button secondary"
                  type="button"
                  onClick={() => {
                    setFile(null);
                    setResp(null);
                    setDatasetError(null);
                    setProgress(0);
                  }}
                  disabled={busy}
                >
                  Clear
                </button>
              </div>
            </div>

            {busy && (
              <div className="section">
                <div className="muted">Uploading... {progress}%</div>
                <div
                  style={{
                    height: "0.6rem",
                    borderRadius: "999px",
                    border: "1px solid rgba(148, 163, 184, 0.3)",
                    overflow: "hidden"
                  }}
                >
                  <div
                    style={{
                      width: `${progress}%`,
                      height: "100%",
                      background: "linear-gradient(135deg, var(--accent), var(--accent-strong))"
                    }}
                  />
                </div>
              </div>
            )}

            {datasetError && (
              <div className="error" style={{ display: "flex", justifyContent: "space-between", gap: "0.8rem" }}>
                <span>{datasetError}</span>
                {authRequired && (
                  <button className="button secondary compact" type="button" onClick={redirectToLogin}>
                    Login
                  </button>
                )}
              </div>
            )}
          </div>

          {resp && (
            <div className="card">
              <div className="section">
                <div className="muted">
                  Saved as <strong>{resp.original_filename}</strong> ({formatBytes(resp.bytes)})
                </div>
                <div>
                  <strong>Columns:</strong> {resp.columns?.join(", ") || "n/a"}
                </div>
                <div>
                  <strong>Preview (first 20 rows)</strong>
                </div>
                <pre
                  style={{
                    padding: "1rem",
                    borderRadius: "0.8rem",
                    border: "1px solid rgba(148, 163, 184, 0.2)",
                    background: "rgba(148, 163, 184, 0.08)",
                    fontSize: "0.85rem",
                    overflow: "auto"
                  }}
                >
                  {JSON.stringify(resp.preview, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </>
      )}

      {activeTab === "library" && (
        <>
          <div>
            <h3>Dataset Library</h3>
            <p className="muted">Browse uploads, set an active dataset, and manage files.</p>
          </div>
          <div className="card">
            <div className="section">
              <div className="toolbar">
                <input
                  placeholder="Filter datasets"
                  value={filter}
                  onChange={(event) => setFilter(event.target.value)}
                />
              </div>

              {datasetsLoading && <div className="empty">Loading datasets...</div>}

              {!datasetsLoading && filteredDatasets.length === 0 && (
                <div className="empty">
                  {datasets.length === 0 ? "No datasets uploaded yet." : "No datasets match the filter."}
                </div>
              )}

              {!datasetsLoading &&
                filteredDatasets.length > 0 &&
                filteredDatasets.map((dataset) => (
                  <div
                    key={dataset.version_id}
                    style={{
                      border: "1px solid rgba(148,163,184,0.25)",
                      borderRadius: "0.8rem",
                      padding: "0.75rem",
                      marginBottom: "0.5rem"
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}>
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontWeight: 600, wordBreak: "break-word" }}>
                          {dataset.original_filename}
                          {dataset.version_id === activeVersionId && (
                            <span className="pill" style={{ marginLeft: "0.5rem" }}>
                              ACTIVE
                            </span>
                          )}
                        </div>
                        <div className="muted" style={{ fontSize: "0.8rem" }}>
                          {formatBytes(dataset.size_bytes)} | {new Date(dataset.uploaded_at).toLocaleString()}
                        </div>
                      </div>
                      <button
                        className="button secondary"
                        type="button"
                        onClick={() => setSelectedVersionId(dataset.version_id)}
                      >
                        Select
                      </button>
                    </div>

                    <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.6rem", flexWrap: "wrap" }}>
                      <button
                        className="button secondary"
                        type="button"
                        onClick={() => {
                          setSelectedVersionId(dataset.version_id);
                          setActiveTab("preview");
                        }}
                      >
                        Preview
                      </button>
                      <button className="button secondary" type="button" onClick={() => downloadDataset(dataset.version_id)}>
                        Download
                      </button>
                      <button
                        className="button"
                        type="button"
                        onClick={() => setActiveDataset(dataset.version_id)}
                        disabled={demoWriteBlocked}
                        title={demoWriteTitle}
                      >
                        Set Active
                      </button>
                      <button
                        className="button secondary"
                        type="button"
                        onClick={() => deleteDataset(dataset.version_id, dataset.original_filename)}
                        disabled={demoWriteBlocked}
                        title={demoWriteTitle}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
            </div>
          </div>
          {datasetError && (
            <div className="error" style={{ display: "flex", justifyContent: "space-between", gap: "0.8rem" }}>
              <span>{datasetError}</span>
              {authRequired && (
                <button className="button secondary compact" type="button" onClick={redirectToLogin}>
                  Login
                </button>
              )}
            </div>
          )}
        </>
      )}
      {activeTab === "preview" && (
        <>
          <div>
            <h3>Preview</h3>
            <p className="muted">Toggle between raw and scored previews for the selected dataset.</p>
          </div>

          <div className="card">
            <div className="toolbar">
              <button
                className={`button${previewMode === "raw" ? "" : " secondary"}`}
                type="button"
                onClick={() => setPreviewMode("raw")}
              >
                Raw Preview
              </button>
              <button
                className={`button${previewMode === "scored" ? "" : " secondary"}`}
                type="button"
                onClick={() => setPreviewMode("scored")}
              >
                Scored Preview
              </button>
              {previewMode === "scored" && (
                <>
                  <label className="field">
                    <span>Threshold</span>
                    <input
                      type="range"
                      min={0.1}
                      max={0.9}
                      step={0.05}
                      value={scoredThreshold}
                      onChange={(event) => setScoredThreshold(Number(event.target.value))}
                    />
                  </label>
                  <span className="pill">>= {scoredThreshold.toFixed(2)}</span>
                  <label className="field">
                    <span>Fraud only</span>
                    <input
                      type="checkbox"
                      checked={scoredFraudOnly}
                      onChange={(event) => setScoredFraudOnly(event.target.checked)}
                    />
                  </label>
                  <label className="field">
                    <span>Limit</span>
                    <select
                      value={scoredLimit}
                      onChange={(event) => setScoredLimit(Number(event.target.value))}
                    >
                      <option value={50}>50</option>
                      <option value={100}>100</option>
                      <option value={200}>200</option>
                    </select>
                  </label>
                </>
              )}
            </div>
            {previewError && <div className="error">{previewError}</div>}
          </div>

          {!selectedVersionId && <div className="empty">Select a dataset from the library.</div>}

          {selectedVersionId && previewMode === "raw" && (
            <DataTable
              title="Raw Preview"
              subtitle={selectedDataset ? selectedDataset.original_filename : undefined}
              columns={previewColumns.length ? previewColumns : ["Loading..."]}
              rows={previewColumns.length ? toTableRows(previewRows, previewColumns) : []}
              loading={previewLoading}
              emptyText="No preview rows."
            />
          )}

          {selectedVersionId && previewMode === "scored" && (
            <>
              <div className="grid">
                <StatCard label="Fraud" value={scoredPreviewSummary?.fraud ?? "n/a"} />
                <StatCard label="Legit" value={scoredPreviewSummary?.legit ?? "n/a"} />
                <StatCard
                  label="Fraud Rate"
                  value={
                    typeof scoredPreviewSummary?.fraud_rate === "number"
                      ? `${(scoredPreviewSummary.fraud_rate * 100).toFixed(1)}%`
                      : "n/a"
                  }
                />
                <StatCard
                  label="Avg Score"
                  value={
                    typeof scoredPreviewSummary?.avg_score === "number"
                      ? scoredPreviewSummary.avg_score.toFixed(3)
                      : "n/a"
                  }
                />
              </div>

              <DataTable
                title="Scored Preview"
                subtitle={selectedDataset ? selectedDataset.original_filename : undefined}
                columns={scoredPreviewTableColumns}
                rows={scoredPreviewTableRows}
                loading={previewLoading}
                emptyText={scoredFraudOnly ? "No fraud rows in preview." : "No scored preview rows."}
              />
            </>
          )}
        </>
      )}

      {activeTab === "quality" && (
        <>
          <div>
            <h3>Quality Profile</h3>
            <p className="muted">Sample-based health checks with schema mapping for critical fields.</p>
          </div>

          {!selectedVersionId && <div className="empty">Select a dataset from the library.</div>}

          {selectedVersionId && (
            <>
              <div className="grid">
                <StatCard label="Sample Rows" value={profile?.sample_rows ?? "n/a"} />
                <StatCard
                  label="Duplicate Estimate"
                  value={profile ? `${profile.duplicate_estimate_percent}%` : "n/a"}
                />
                <StatCard label="Invalid Timestamps" value={profile?.invalid_timestamp_count ?? "n/a"} />
                <StatCard label="Mapping" value={profile?.mapping_source ?? "n/a"} />
              </div>

              {profileError && <div className="error">{profileError}</div>}

              {profile && (
                <>
                  {mappingWarnings.length > 0 && (
                    <div className="error-banner">
                      Missing mappings: {mappingWarnings.join(", ")}. Scoring quality improves when mapped.
                    </div>
                  )}
                  <div className="card">
                    <div className="section">
                      <div style={{ fontWeight: 600 }}>Schema Mapping</div>
                      <div className="form-grid" style={{ marginTop: "0.8rem" }}>
                        <label className="field">
                          <span>Amount</span>
                          <select
                            value={schemaMapping.amount_col ?? ""}
                            onChange={(event) => updateMapping("amount_col", event.target.value)}
                          >
                            <option value="">Unmapped</option>
                            {profile.columns.map((col) => (
                              <option key={`amount-${col}`} value={col}>
                                {col}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="field">
                          <span>Timestamp</span>
                          <select
                            value={schemaMapping.timestamp_col ?? ""}
                            onChange={(event) => updateMapping("timestamp_col", event.target.value)}
                          >
                            <option value="">Unmapped</option>
                            {profile.columns.map((col) => (
                              <option key={`ts-${col}`} value={col}>
                                {col}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="field">
                          <span>User ID</span>
                          <select
                            value={schemaMapping.user_id_col ?? ""}
                            onChange={(event) => updateMapping("user_id_col", event.target.value)}
                          >
                            <option value="">Unmapped</option>
                            {profile.columns.map((col) => (
                              <option key={`user-${col}`} value={col}>
                                {col}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="field">
                          <span>Merchant</span>
                          <select
                            value={schemaMapping.merchant_col ?? ""}
                            onChange={(event) => updateMapping("merchant_col", event.target.value)}
                          >
                            <option value="">Unmapped</option>
                            {profile.columns.map((col) => (
                              <option key={`merchant-${col}`} value={col}>
                                {col}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="field">
                          <span>Device ID</span>
                          <select
                            value={schemaMapping.device_id_col ?? ""}
                            onChange={(event) => updateMapping("device_id_col", event.target.value)}
                          >
                            <option value="">Unmapped</option>
                            {profile.columns.map((col) => (
                              <option key={`device-${col}`} value={col}>
                                {col}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="field">
                          <span>Country</span>
                          <select
                            value={schemaMapping.country_col ?? ""}
                            onChange={(event) => updateMapping("country_col", event.target.value)}
                          >
                            <option value="">Unmapped</option>
                            {profile.columns.map((col) => (
                              <option key={`country-${col}`} value={col}>
                                {col}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="field">
                          <span>Label (optional)</span>
                          <select
                            value={schemaMapping.label_col ?? ""}
                            onChange={(event) => updateMapping("label_col", event.target.value)}
                          >
                            <option value="">Unmapped</option>
                            {profile.columns.map((col) => (
                              <option key={`label-${col}`} value={col}>
                                {col}
                              </option>
                            ))}
                          </select>
                        </label>
                      </div>
                      <div className="toolbar" style={{ marginTop: "0.8rem" }}>
                        <button
                          className="button"
                          type="button"
                          onClick={saveSchemaMapping}
                          disabled={demoWriteBlocked || mappingSaving}
                          title={demoWriteTitle}
                        >
                          {mappingSaving ? "Saving..." : "Save Mapping"}
                        </button>
                        {mappingMessage && <div className="success-banner">{mappingMessage}</div>}
                        {mappingError && <div className="error">{mappingError}</div>}
                      </div>
                    </div>
                  </div>
                  <DataTable
                    title="Missing by Column"
                    columns={["Column", "Missing %"]}
                    rows={Object.entries(profile.missing_by_column).map(([column, value]) => [
                      column,
                      `${value}%`
                    ])}
                    loading={profileLoading}
                    emptyText="No missing data stats."
                  />
                  <DataTable
                    title="Amount-like Stats"
                    columns={["Column", "Min", "Max", "Mean", "P95"]}
                    rows={Object.entries(profile.numeric_stats).map(([column, stats]) => [
                      column,
                      stats.min,
                      stats.max,
                      stats.mean,
                      stats.p95
                    ])}
                    loading={profileLoading}
                    emptyText="No numeric stats."
                  />
                </>
              )}
            </>
          )}
        </>
      )}

      {activeTab === "jobs" && (
        <>
          <div>
            <h3>Scoring Jobs</h3>
            <p className="muted">Launch batch scoring and monitor progress.</p>
          </div>

          <div className="card">
            <div className="toolbar">
              <label className="field">
                <span>Threshold</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={jobThreshold}
                  onChange={(event) => setJobThreshold(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Model version (optional)</span>
                <input
                  value={jobModelVersion}
                  onChange={(event) => setJobModelVersion(event.target.value)}
                  placeholder="demo-v1"
                />
              </label>
              <button
                className="button"
                type="button"
                onClick={startJob}
                disabled={demoWriteBlocked}
                title={demoWriteTitle}
              >
                Start Scoring Job
              </button>
              <button className="button secondary" type="button" onClick={fetchJobs}>
                Refresh
              </button>
            </div>
            {jobsError && <div className="error">{jobsError}</div>}
          </div>

          <DataTable
            title="Jobs"
            columns={["Job", "Dataset", "Status", "Progress", "Threshold", "Updated", "Actions"]}
            rows={jobs.map((job) => {
              const progress = job.rows_total
                ? `${job.rows_done}/${job.rows_total}`
                : `${job.rows_done}`;
              const dataset = datasets.find((item) => item.version_id === job.dataset_version_id);
              return [
                job.job_id.slice(0, 8),
                dataset?.original_filename ?? "n/a",
                job.status,
                progress,
                job.threshold ?? "n/a",
                new Date(job.updated_at).toLocaleString(),
                <div key={job.job_id} style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
                  <button className="button secondary" type="button" onClick={() => setSelectedJobId(job.job_id)}>
                    Use
                  </button>
                  {job.output_path && (
                    <button
                      className="button secondary"
                      type="button"
                      onClick={() =>
                        window.open(`${API_BASE}/scoring-jobs/${encodeURIComponent(job.job_id)}/download`, "_blank")
                      }
                    >
                      Download
                    </button>
                  )}
                </div>
              ];
            })}
            loading={jobsLoading}
            emptyText="No scoring jobs yet."
          />
        </>
      )}
      {activeTab === "results" && (
        <>
          <div>
            <h3>Fraud Results</h3>
            <p className="muted">Explore scored rows, filter by threshold, and review top segments.</p>
          </div>

          <div className="card">
            <div className="toolbar">
              <label className="field">
                <span>Job</span>
                <select value={selectedJobId ?? ""} onChange={(event) => setSelectedJobId(event.target.value)}>
                  <option value="" disabled>
                    Select job
                  </option>
                  {jobs.map((job) => (
                    <option key={job.job_id} value={job.job_id}>
                      {job.job_id.slice(0, 8)} ({job.status})
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Threshold</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={resultsThreshold}
                  onChange={(event) => setResultsThreshold(Number(event.target.value))}
                />
              </label>
              <span className="pill">>= {resultsThreshold.toFixed(2)}</span>
              <label className="field">
                <span>Fraud only</span>
                <input
                  type="checkbox"
                  checked={fraudOnly}
                  onChange={(event) => setFraudOnly(event.target.checked)}
                />
              </label>
              <button
                className="button secondary"
                type="button"
                onClick={() => selectedJobId && fetchResults(selectedJobId, resultsOffset, resultsLimit, fraudOnly)}
              >
                Refresh
              </button>
            </div>
            {resultsError && <div className="error">{resultsError}</div>}
            {results?.is_partial && (
              <div
                style={{
                  padding: "0.8rem 1rem",
                  borderRadius: "0.9rem",
                  background: "rgba(250, 204, 21, 0.2)",
                  color: "#92400e",
                  fontWeight: 600
                }}
              >
                Job running — showing partial results.
              </div>
            )}
            {feedbackMessage && <div className="success-banner">{feedbackMessage}</div>}
            {feedbackError && <div className="error">{feedbackError}</div>}
          </div>

          <div className="card">
            <div className="section">
              <div style={{ fontWeight: 600 }}>Risk Distribution</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "0.6rem" }}>
                {riskBins.map((count, index) => (
                  <div key={`bin-${index}`} style={{ textAlign: "center" }}>
                    <div
                      style={{
                        height: "60px",
                        borderRadius: "0.6rem",
                        background: "rgba(14, 165, 233, 0.15)",
                        display: "flex",
                        alignItems: "flex-end",
                        justifyContent: "center",
                        padding: "0.4rem"
                      }}
                    >
                      <div
                        style={{
                          height: `${Math.min(100, count * 12)}%`,
                          width: "100%",
                          borderRadius: "0.5rem",
                          background: "linear-gradient(135deg, var(--accent), var(--accent-strong))"
                        }}
                      />
                    </div>
                    <div className="muted" style={{ fontSize: "0.75rem", marginTop: "0.4rem" }}>
                      {count} rows
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="section">
              <div style={{ fontWeight: 600 }}>Top Segments</div>
              <div className="grid">
                {topSegments.map((segment) => (
                  <div key={segment.label} className="card" style={{ padding: "1rem" }}>
                    <div style={{ fontWeight: 600 }}>{segment.label}</div>
                    {segment.top.length === 0 && <div className="muted">No data</div>}
                    {segment.top.map(([label, count]) => (
                      <div key={label} className="muted" style={{ fontSize: "0.85rem" }}>
                        {label}: {count}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <DataTable
            title="Scored Rows"
            columns={resultTableColumns}
            rows={resultTableRows}
            loading={resultsLoading}
            emptyText="No scored rows."
            footer={
              <div className="table-footer">
                <div className="muted">
                  Showing {displayedResults.length} rows (offset {resultsOffset})
                </div>
                <div className="table-pagination">
                  <button
                    className="button secondary"
                    type="button"
                    onClick={() => setResultsOffset(Math.max(0, resultsOffset - resultsLimit))}
                    disabled={resultsOffset === 0}
                  >
                    Prev
                  </button>
                  <button
                    className="button secondary"
                    type="button"
                    onClick={() => results?.has_more && setResultsOffset(resultsOffset + resultsLimit)}
                    disabled={!results?.has_more}
                  >
                    Next
                  </button>
                </div>
              </div>
            }
          />
        </>
      )}

      {activeTab === "cases" && (
        <>
          <div>
            <h3>Create Cases</h3>
            <p className="muted">Create cases from selected fraud rows and capture analyst feedback.</p>
          </div>

          <div className="card">
            <div className="toolbar">
              <span className="pill">Selected: {selectedTxIds.size}</span>
              <button
                className="button"
                type="button"
                onClick={createCases}
                disabled={demoWriteBlocked}
                title={demoWriteTitle}
              >
                Create Cases
              </button>
              <button className="button secondary" type="button" onClick={clearSelection}>
                Clear Selection
              </button>
            </div>
            {caseMessage && (
              <div className="success-banner">
                {caseMessage}
                {createdCaseId && (
                  <span style={{ marginLeft: "0.75rem" }}>
                    <a href={`/cases/${createdCaseId}`}>View case</a>
                  </span>
                )}
              </div>
            )}
            {caseError && <div className="error">{caseError}</div>}
          </div>

          <div className="card">
            <div className="toolbar">
              <label className="field">
                <span>Feedback label</span>
                <select value={feedbackLabel} onChange={(event) => setFeedbackLabel(event.target.value as "FRAUD" | "LEGIT")}>
                  <option value="FRAUD">FRAUD</option>
                  <option value="LEGIT">LEGIT</option>
                </select>
              </label>
              <button
                className="button secondary"
                type="button"
                onClick={submitFeedback}
                disabled={demoWriteBlocked}
                title={demoWriteTitle}
              >
                Submit Feedback
              </button>
            </div>
            {feedbackMessage && <div className="success-banner">{feedbackMessage}</div>}
            {feedbackError && <div className="error">{feedbackError}</div>}
          </div>

          {selectedTxIds.size > 0 && (
            <div className="card">
              <div className="section">
                <div style={{ fontWeight: 600 }}>Selected Transactions</div>
                <div className="muted" style={{ fontSize: "0.85rem" }}>
                  {Array.from(selectedTxIds).slice(0, 15).join(", ")}
                  {selectedTxIds.size > 15 && " ..."}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {activeTab === "audit" && (
        <>
          <div>
            <h3>Audit</h3>
            <p className="muted">Recent dataset actions, scoring events, and downloads.</p>
          </div>

          <DataTable
            title="Audit Entries"
            columns={["Timestamp", "Actor", "Action", "Decision", "Model", "Score"]}
            rows={auditEntries.map((entry) => [
              new Date(entry.timestamp).toLocaleString(),
              entry.actor ?? "system",
              entry.action,
              entry.decision,
              `${entry.model_name}:${entry.model_version}`,
              entry.score
            ])}
            loading={auditLoading}
            emptyText="No audit entries yet."
          />
          {auditError && <div className="error">{auditError}</div>}
        </>
      )}
    </section>
  );
}
