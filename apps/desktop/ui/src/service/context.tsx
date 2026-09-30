import { createContext, useContext, type JSX } from "solid-js";
import type { CerebroService } from "./service";

const ServiceContext = createContext<CerebroService>();

export function ServiceProvider(props: { service: CerebroService; children: JSX.Element }): JSX.Element {
  return <ServiceContext.Provider value={props.service}>{props.children}</ServiceContext.Provider>;
}

export function useService(): CerebroService {
  const service = useContext(ServiceContext);
  if (!service) throw new Error("useService() called outside <ServiceProvider>");
  return service;
}
